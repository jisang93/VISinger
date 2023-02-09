# Based on https://github.com/NATSpeech/NATSpeech
import json
import re

from g2pk import G2p  # Docker can't install G2p. Do not use this library in the docker.
from jamo import h2j

from preprocessor.text.base_text_processor import BaseTextProcessor, register_text_processors
from utils.commons.hparams import hparams
from utils.text.text_encoder import PUNCS


@register_text_processors("ko_sing")
class KoreanSingingProcessor(BaseTextProcessor):
    # G2pk settings
    g2p = G2p()
    # Dictionary settings
    dictionary = json.load(open("./preprocessor/text/dict/korean.json", "r"))
    num_checker = "([+-]?\d{1,3},\d{3}(?!\d)|[+-]?\d+)[\.]?\d*"
    PUNCS += ",\'\""

    @staticmethod
    def sp_phonemes():
        return ['|']

    @classmethod
    def preprocess_text(cls, text):
        # Normalize basic pattern
        text = text.strip()
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z가-힣]", "", text)
        text = re.sub(f" ?([{cls.PUNCS}]) ?", r"\1", text)  # !! -> !
        text = re.sub(f"([{cls.PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub('\(\d+일\)', '', text)
        text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)
        text = re.sub(f"([{cls.PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r" ", text)
        # Normalize with prepared dictionaries
        text = cls.normalize_with_dictionary(text, cls.dictionary["etc_dict"])
        text = cls.normalize_english(text, cls.dictionary["eng_dict"])
        text = cls.normalize_upper(text, cls.dictionary["upper_dict"])
        # number to hanguel
        text = cls.normalize_number(text, cls.num_checker, cls.dictionary)
        return text

    @staticmethod
    def normalize_with_dictionary(text, dictionary):
        """ Check special korean pronounciation in dictionary """
        if any(key in text for key in dictionary.keys()):
            pattern = re.compile("|".join(re.escape(key) for key in dictionary.keys()))
            return pattern.sub(lambda x: dictionary[x.group()], text)
        else:
            return text
    
    @staticmethod
    def normalize_english(text, dictionary):
        """ Convert English to Korean pronounciation """
        def _eng_replace(w):
            word = w.group()
            if word in dict:
                return dictionary[word]
            else:
                return word
        text = re.sub("([A-Za-z]+)", _eng_replace, text)
        return text

    @staticmethod
    def normalize_upper(text, dictionary):
        """ Convert lower English to Upper English and Changing to Korean pronounciation"""
        def upper_replace(w):
            word = w.group()
            if all([char.isupper() for char in word]):
                return "".join(dictionary[char] for char in word)
            else:
                return word
        text = re.sub("[A-Za-z]+", upper_replace, text)

        return text

    @classmethod
    def normalize_number(cls, text, num_checker, dictionary):
        """ Convert Numbert to Korean pronounciation """
        text = cls.normalize_with_dictionary(text, dictionary["unit_dict"])
        text = re.sub(num_checker + dictionary["count_checker"],
                      lambda x: cls.num_to_hangeul(x, dictionary, True), text)
        text = re.sub(num_checker,
                      lambda x: cls.num_to_hangeul(x, dictionary, False), text)
        return text

    @staticmethod
    def num_to_hangeul(num_str, dictionary, is_count=False):
        """ Following https://github.com/keonlee9420/Expressive-FastSpeech2/blob/main/text/korean.py
        Normalize number prounciation. """
        zero_cnt = 0
        # Check Korean count unit
        if is_count:
            num_str, unit_str = num_str.group(1), num_str.group(2)
        else:
            num_str, unit_str = num_str.group(), ""
        # Remove decimal separator
        num_str = num_str.replace(",", "")

        if is_count and len(num_str) > 2:
            is_count = False
        
        if len(num_str) > 1 and num_str.startwith("0") and "." not in num_str:
            for n in num_str:
                zero_cnt += 1 if n == "0" else 0
            num_str = num_str[zero_cnt:]
        
        kor = ""
        if num_str != "":
            if num_str == "0":
                return "영 " + (unit_str if unit_str else "")
            # Split float number
            check_float = num_str.split(".")
            if len(check_float) == 2:
                digit_str, float_str = check_float
            elif len(check_float) >= 3:
                raise Exception(f"| Wrong number format: {num_str}")
            else:
                digit_str, float_str = check_float[0], None
            if is_count and float_str is not None:
                raise Exception(f"| 'is_count' and float number does not fit each other")
            # Check minus or plus symbol
            digit = int(digit_str)
            if digit_str.startswith("-") or digit_str.startswith("+"):
                digit, digit_str = abs(digit), str(abs(digit))
            size = len(str(digit))
            tmp = []
            for i, v in enumerate(digit_str, start=1):
                v = int(v)
                if v != 0:
                    if is_count:
                        tmp += dictionary["count_dict"][v]
                    else:
                        tmp += dictionary["num_dict"][str(v)]
                        if v == 1 and i != 1 and i != len(digit_str):
                            tmp = tmp[:-1]
                    tmp += dictionary["num_ten_dict"][(size - i) % 4]
                if (size - i) % 4 == 0 and len(tmp) != 0:
                    kor += "".join(tmp)
                    tmp = []
                    kor += dictionary["num_tenthousand_dict"][int((size - i) / 4)]
            if is_count:
                if kor.startswith("한") and len(kor) > 1:
                    kor = kor[1:]
                
                if any(word in kor for word in dictionary["count_tenth_dict"]):
                    kor = re.sub("|".join(dictionary["count_tenth_dict"].keys()),
                                 lambda x: dictionary["count_tenth_dict"][x.group()], kor)
            if not is_count and kor.startswith("일") and len(kor) > 1:
                kor = kor[1:]
            if float_str is not None and float_str != "":
                kor += "영" if kor == "" else ""
                kor += "쩜 "
                kor += re.sub("\d", lambda x: dictionary["num_dict"][x.group()], float_str)
            if num_str.startswith("+"):
                kor = "플러스 " + kor
            elif num_str.startswith("-"):
                kor = "마이너스 " + kor
            if zero_cnt > 0:
                kor = "공" * zero_cnt + kor
            return kor + unit_str

    @classmethod
    def process(cls, midi_info, hparams):
        midi_info_ = []
        ph_list = []
        n_frame = hparams["preprocess_args"]["num_frame"]
        sr = hparams["sample_rate"]
        hop_size = hparams["hop_size"]
        frame_time = n_frame * hop_size / sr
        text = "".join([midi[7] for midi in midi_info])
        text = [cls.g2p(word) for word in text.split("|")]
        text = "|".join(text)
        assert len(text) == len(midi_info), f"| Wrong text process: {len(text)}, {len(midi_info)}"
        # Korean singing voice processing
        for i, (bar, pos, pitch, duration, start_time, end_time, tempo, _) in enumerate(midi_info):
            phs = h2j(cls.preprocess_text(text[i]))
            ph = [p for p in phs if p != " " or p != ""] if len(phs) != 0 else ["|"]
            if len(ph) == 1:
                notes = [[bar, pos, pitch, duration, start_time, end_time, tempo, ph]]
            elif len(ph) == 2:
                notes = []
                if int((end_time - start_time) * sr / hop_size + 0.5) > n_frame:
                    for j, p in enumerate(ph):
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + frame_time, tempo, p]
                        else:
                            note = [bar, pos, pitch, duration, start_time + frame_time, end_time, tempo, p]
                        notes.append(note)
                else:
                    except_frame_time = (n_frame - 2) * hop_size / sr
                    for j, p in enumerate(ph):
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + except_frame_time, tempo, p]
                        else:
                            note = [bar, pos, pitch, duration, start_time + except_frame_time, end_time, tempo, p]
                        notes.append(note)
            elif len(ph) == 3:
                notes = []
                if int((end_time - start_time) * sr / hop_size + 0.5) >= n_frame * 3:
                    for j, p in enumerate(ph):
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + frame_time, tempo, p]
                        elif j == 1:
                            note = [bar, pos, pitch, duration, start_time + frame_time, end_time - frame_time, tempo, p]
                        else:
                            note = [bar, pos, pitch, duration, end_time - frame_time, end_time, tempo, p]
                        notes.append(note)
                elif int((end_time - start_time) * sr / hop_size + 0.5) >= n_frame * 2:
                    except_frame_time = (n_frame - 1) * hop_size / sr
                    for j, p in enumerate(ph):
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + except_frame_time, tempo, p]
                        elif j == 1:
                            note = [bar, pos, pitch, duration, start_time + except_frame_time, end_time - except_frame_time, tempo, p]
                        else:
                            note = [bar, pos, pitch, duration, end_time - except_frame_time, end_time, tempo, p]
                        notes.append(note)
                elif int((end_time - start_time) * sr / hop_size + 0.5) >= n_frame:
                    except_frame_time = (n_frame - 2) * hop_size / sr
                    for j, p in enumerate(ph):
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + except_frame_time, tempo, p]
                        elif j == 1:
                            note = [bar, pos, pitch, duration, start_time + except_frame_time, end_time - except_frame_time, tempo, p]
                        else:
                            note = [bar, pos, pitch, duration, end_time - except_frame_time, end_time, tempo, p]
                        notes.append(note)
                else:
                    for j, p in enumerate(ph):
                        except_frame_time = (n_frame - 2) * hop_size / sr
                        if j == 0:
                            note = [bar, pos, pitch, duration, start_time, start_time + 1, tempo, p]
                        elif j == 1:
                            note = [bar, pos, pitch, duration, start_time + 1, end_time - 1, tempo, p]
                        elif j == 2:
                            note = [bar, pos, pitch, duration, end_time - 1, end_time, tempo, p]
                        notes.append(note)
            assert len(ph) == len(notes), f"| Wrong settings: ph = {len(ph)}, notes = {len(notes)}"
            ph_list.extend(ph)
            midi_info_.extend(notes)
        return ph_list, midi_info_
