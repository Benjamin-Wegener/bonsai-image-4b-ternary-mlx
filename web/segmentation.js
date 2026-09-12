/**
 * Semantic Mask Segmentation, Region Palette & Magic Wand UI Logic
 */

let segmentationMapImg = new Image();
let segmentationData = null;
let detectedSegments = [];
let showSegOverlay = false;

async function runAiModelSegmentation() {
    if (!base64Ref) {
        document.getElementById('segStatusLabel').innerText = "⚠️ Please load a reference image first!";
        return;
    }
    const steps = 8;
    const btn = document.getElementById('btnRunSeg');
    btn.disabled = true;
    btn.innerText = `⏳ Segmenting...`;
    document.getElementById('segStatusLabel').innerText = `Running Object Mask Segmentation...`;

    try {
        const res = await fetch('/v1/segment_mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Ref, num_regions: 8, use_model: true, steps: steps })
        });

        if (!res.ok) throw new Error("Segmentation server returned status " + res.status);
        const data = await res.json();

        segmentationMapImg.onload = () => {
            const tempCanv = document.createElement('canvas');
            tempCanv.width = mainCanvas.width;
            tempCanv.height = mainCanvas.height;
            const tc = tempCanv.getContext('2d', { willReadFrequently: true });
            tc.imageSmoothingEnabled = false;
            tc.drawImage(segmentationMapImg, 0, 0, mainCanvas.width, mainCanvas.height);
            segmentationData = tc.getImageData(0, 0, mainCanvas.width, mainCanvas.height);

            detectedSegments = data.segments.map(s => ({
                ...s,
                selected: false
            }));
            renderSegmentChips();
            document.getElementById('segRegionsContainer').style.display = 'block';
            document.getElementById('segStatusLabel').innerText = `Detected ${data.segments.length} object regions (click to select)`;
            
            showSegOverlay = false;
            const btn = document.getElementById('btnToggleSegOverlay');
            if (btn) {
                btn.classList.remove('active');
                btn.style.opacity = '0.5';
                btn.innerText = '🎨 Segment Colors';
            }
            renderLiveComposite(false);
        };
        segmentationMapImg.src = "data:image/png;base64," + data.segmentation_b64;
    } catch (err) {
        console.warn("Segmentation Error:", err);
        document.getElementById('segStatusLabel').innerText = "⚠️ Segmentation failed: " + err.message;
    } finally {
        btn.disabled = false;
        btn.innerText = "✂️ Segment";
    }
}

function renderSegmentChips() {
    const list = document.getElementById('segChipsList');
    if (!list) return;
    list.innerHTML = '';

    detectedSegments.forEach((seg, idx) => {
        const div = document.createElement('div');
        div.className = `ps-seg-chip ${seg.selected ? 'selected' : ''}`;
        div.onclick = () => toggleSegment(idx);

        div.innerHTML = `
            <div style="display:flex; align-items:center;">
                <span class="ps-color-swatch" style="background:${seg.color};"></span>
                <span style="font-weight:600; color:#fff;">${seg.label}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <span style="color:var(--ps-text-dim); font-size:10px;">${seg.percentage}%</span>
                <input type="checkbox" ${seg.selected ? 'checked' : ''} style="accent-color:var(--ps-cyan); pointer-events:none;">
            </div>
        `;
        list.appendChild(div);
    });
}

function toggleSegment(idx) {
    saveMaskUndoState();
    detectedSegments[idx].selected = !detectedSegments[idx].selected;
    renderSegmentChips();
    rebuildMaskFromSegments();
}

function selectAllSegments(select) {
    saveMaskUndoState();
    detectedSegments.forEach(s => s.selected = select);
    renderSegmentChips();
    rebuildMaskFromSegments();
}

function invertAllSegments() {
    saveMaskUndoState();
    detectedSegments.forEach(s => s.selected = !s.selected);
    renderSegmentChips();
    rebuildMaskFromSegments();
}

function handleWandClick(x, y) {
    if (!segmentationData) {
        document.getElementById('segStatusLabel').innerText = "⚠️ Please run Object Mask Segmentation first to use Magic Wand!";
        return;
    }
    const px = Math.floor(x);
    const py = Math.floor(y);
    const w = mainCanvas.width;
    const h = mainCanvas.height;
    if (px < 0 || px >= w || py < 0 || py >= h) return;

    const idx = (py * w + px) * 4;
    const r = segmentationData.data[idx];
    const g = segmentationData.data[idx + 1];
    const b = segmentationData.data[idx + 2];

    let matchedIdx = -1;
    let minDist = 99999999;

    detectedSegments.forEach((seg, sIdx) => {
        const dr = seg.rgb[0] - r;
        const dg = seg.rgb[1] - g;
        const db = seg.rgb[2] - b;
        const dist = dr * dr + dg * dg + db * db;
        if (dist < minDist) {
            minDist = dist;
            matchedIdx = sIdx;
        }
    });

    if (matchedIdx !== -1) {
        toggleSegment(matchedIdx);
    }
}

function rebuildMaskFromSegments() {
    const w = mainCanvas.width;
    const h = mainCanvas.height;
    if (!segmentationData) return;

    // Ensure segmentationData matches current canvas dimensions
    if (segmentationData.width !== w || segmentationData.height !== h) {
        const tempCanv = document.createElement('canvas');
        tempCanv.width = w;
        tempCanv.height = h;
        const tc = tempCanv.getContext('2d', { willReadFrequently: true });
        tc.imageSmoothingEnabled = false;
        tc.drawImage(segmentationMapImg, 0, 0, w, h);
        segmentationData = tc.getImageData(0, 0, w, h);
    }

    const maskImgData = maskCtx.createImageData(w, h);
    const segData = segmentationData.data;
    const selectedIndices = new Set(
        detectedSegments
            .map((s, idx) => s.selected ? idx : -1)
            .filter(idx => idx !== -1)
    );

    if (selectedIndices.size > 0) {
        const segRGBs = detectedSegments.map(s => s.rgb);
        for (let i = 0; i < segData.length; i += 4) {
            const r = segData[i];
            const g = segData[i + 1];
            const b = segData[i + 2];

            let closestIdx = 0;
            let minDist = 99999999;
            for (let sIdx = 0; sIdx < segRGBs.length; sIdx++) {
                const dr = segRGBs[sIdx][0] - r;
                const dg = segRGBs[sIdx][1] - g;
                const db = segRGBs[sIdx][2] - b;
                const dist = dr * dr + dg * dg + db * db;
                if (dist < minDist) {
                    minDist = dist;
                    closestIdx = sIdx;
                }
            }

            if (selectedIndices.has(closestIdx)) {
                maskImgData.data[i] = 255;
                maskImgData.data[i + 1] = 255;
                maskImgData.data[i + 2] = 255;
                maskImgData.data[i + 3] = 255;
            } else {
                maskImgData.data[i + 3] = 0;
            }
        }
        maskCtx.putImageData(maskImgData, 0, 0);
    } else {
        maskCtx.clearRect(0, 0, w, h);
    }

    renderLiveComposite(false);
}

function toggleSegOverlay() {
    showSegOverlay = !showSegOverlay;
    const btn = document.getElementById('btnToggleSegOverlay');
    if (btn) {
        btn.classList.toggle('active', showSegOverlay);
        btn.style.opacity = showSegOverlay ? '1.0' : '0.5';
        btn.innerText = showSegOverlay ? '🎨 Segment Colors' : '🚫 Colors Hidden';
    }
    renderLiveComposite(false);
}
