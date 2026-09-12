#!/bin/bash
set -e

echo "1. Sichere alten fehlerhaften .venv Ordner..."
rm -rf /Users/user/dev/Bonsai-Image-Demo/.venv.bak
if [ -d /Users/user/dev/Bonsai-Image-Demo/.venv ]; then
    mv /Users/user/dev/Bonsai-Image-Demo/.venv /Users/user/dev/Bonsai-Image-Demo/.venv.bak
fi

echo "2. Kopiere voll funktionierenden .venv direkt vom Mac mini herüber..."
rsync -aP --delete 192.168.178.71:/Users/user/dev/Bonsai-Image-Demo/.venv/ /Users/user/dev/Bonsai-Image-Demo/.venv/

echo "3. Kopiere App & Vendor Anpassungen herüber..."
rsync -aP 192.168.178.71:/Users/user/dev/Bonsai-Image-Demo/app/ /Users/user/dev/Bonsai-Image-Demo/app/
rsync -aP 192.168.178.71:/Users/user/dev/Bonsai-Image-Demo/server_12345.py /Users/user/dev/Bonsai-Image-Demo/server_12345.py

echo "4. Pfad-Anpassung in mflux.pth..."
echo "/Users/user/dev/Bonsai-Image-Demo/vendor/mflux-prism/src" > /Users/user/dev/Bonsai-Image-Demo/.venv/lib/python3.12/site-packages/mflux.pth

echo "5. Kopiere ggf. fehlende Python Runtime Referenzen..."
if [ ! -d /Users/user/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none ]; then
    mkdir -p /Users/user/.local/share/uv/python
    rsync -aP 192.168.178.71:/Users/user/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/ /Users/user/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/
fi

echo "6. Teste lokalen Import auf diesem Mac..."
/Users/user/dev/Bonsai-Image-Demo/.venv/bin/python -c "
from backend.pipeline import FluxPipeline, PipelineConfig
print('Bonsai Pipeline läuft lokal auf DIESEM Mac!')
"
