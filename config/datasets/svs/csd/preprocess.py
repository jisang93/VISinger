# Based on https://github.com/NATSpeech/NATSpeech
import glob
import os
import miditoolkit
import traceback

from preprocessor.base_preprocessor import BasePreprocessor


class CSDPreprocessor(BasePreprocessor):
    """ We alredy split the note, lyrics and waveform before preprocessing.
        In this repository, we only use Korean singing voice data in CSD dataset."""
    def meta_data(self):
        # Get data
        base_dir = f"{self.raw_data_dir}"
        file_dirs = glob.glob(f"{base_dir}/midi/*.mid")
        # Get song information
        for dir in file_dirs:
            filename = os.path.basename(dir)
            item_name = filename.split(".")[0]
            spk_name = "csd"
            # file directory
            wav_fn = f"{base_dir}/wav/{item_name}.wav"
            with open(f"{base_dir}/text/{item_name}.txt", "r") as f:
                text = f.readline().strip().replace(" ", "")
            # Refine midi file
            try:
                midi_obj = miditoolkit.midi.parser.MidiFile(dir)
                midi_obj = self.refine_midi_file(midi_obj, text)
                yield {"item_name": item_name, "wav_fn": wav_fn, "midi_fn": dir, "midi_obj": midi_obj,
                        "text": text, "spk_name": spk_name}
            except:
                traceback.print_exc()
                print(f"| Error is caught. item_name: {item_name}.")
                pass

    @staticmethod
    def refine_midi_file(midi_obj, lyrics):
        notes = midi_obj.instruments[0].notes
        assert len(notes) == len(lyrics), f"| Note: {len(notes)}, lyrics: {len(lyrics)}"
        lyric_list = []
        for i, lyr in enumerate(lyrics):
            lyric = miditoolkit.Lyric(lyr, notes[i].start)
            lyric_list.append(lyric)
        
        midi_obj.lyrics = lyric_list
        return midi_obj
