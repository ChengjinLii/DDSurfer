"""Generate the upstream xxd-compatible embedded help arrays without xxd."""
from pathlib import Path
import re
import sys


def generate(source, destination):
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    symbol = re.sub(r'[^A-Za-z0-9_]', '_', source.name)
    rows = [', '.join('0x%02x' % value for value in data[start:start + 12])
            for start in range(0, len(data), 12)]
    text = 'unsigned char %s[] = {\n  %s\n};\nunsigned int %s_len = %d;\n' % (
        symbol, ',\n  '.join(rows), symbol, len(data))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


if __name__ == '__main__':
    generate(*sys.argv[1:])
