import re

from app.memory import memory_line


def test_memory_line_has_program_and_system_numbers():
    line = memory_line("test")
    assert line.startswith("[memory, test] program ")
    assert re.search(r"program \d+ MB now, peak \d+ MB \| PC: \d+\.\d GB available of \d+\.\d GB "
                     r"\(\d+ % used\)$", line)
