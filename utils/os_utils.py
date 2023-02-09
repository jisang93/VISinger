# Based on https://github.com/NATSpeech/NATSpeech
import subprocess

from inspect import isfunction


def link_file(from_file, to_file):
    subprocess.check_call(
        f'ln -s "`realpath --relative-to="{to_file}" "{from_file}"`" "{to_file}"', shell=True)


def move_file(from_file, to_file):
    """ Move from_file to to_file. """
    subprocess.check_call(f'mv "{from_file}" "{to_file}"', shell=True)


def copy_file(from_file, to_file):
    """ Copy from_file to to_file. """
    subprocess.check_call(f'cp -r "{from_file}" "{to_file}"', shell=True)


def remove_file(*fns):
    """ Remove files from fns. """
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d
