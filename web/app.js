/**
 * Photoshop Studio Canvas Drawing, Layers, Mask Compositing & Generation App
 */

let base64Ref = null;
let progressPollInterval = null;
let isDrawing = false;
let activeTool = 'brush'; // 'brush', 'eraser', 'wand', 'fill'

let mainCanvas, mainCtx;
let maskCanvas, maskCtx;

let refImgObj = new Image();
let latestGeneratedImgObj = new Image();
let latestGeneratedB64 = null;
let availableLoras = [];

const maskUndoStack = [];
const MAX_UNDO = 30;

let rawRefWidth = 0;
let rawRefHeight = 0;
let croppedGridWidth = 0;
let croppedGridHeight = 0;

// ── Prompt Suggestions: 20 txt2img + 20 img2img/reference ──
const PROMPTS_TXT2IMG = [
    "cinematic portrait of a young woman, soft natural light, 85mm lens, bokeh, ultra detailed skin texture",
    "futuristic cyberpunk city at night, neon lights, rain reflections, cinematic composition, ultra detailed",
    "cozy mountain cabin in autumn forest, warm light from windows, misty morning, photorealistic, 8k",
    "minimal Scandinavian living room interior, natural wood, large windows, daylight, architectural photography",
    "majestic lion in savanna, golden hour, dust particles, National Geographic style, sharp focus",
    "delicious sushi platter, top-down food photography, studio lighting, ultra sharp, 4k",
    "vintage sports car on coastal road, sunset, motion blur, film grain, highly detailed",
    "ethereal forest with glowing mushrooms, fantasy art, volumetric light, highly detailed",
    "astronaut standing on alien planet, two moons in sky, cinematic sci-fi, ultra realistic",
    "Japanese zen garden, raked sand, stone lantern, cherry blossoms, tranquil, photorealistic",
    "high fashion model on runway, dramatic studio lighting, Vogue editorial, sharp focus",
    "underwater coral reef, tropical fish, sun rays through water, ultra detailed, vibrant colors",
    "abandoned industrial warehouse, dramatic light beams, dust, cinematic, photorealistic",
    "cute corgi puppy in meadow, shallow depth of field, sunny day, ultra detailed fur",
    "modern glass skyscraper, blue hour, reflections, architectural photography, ultra sharp",
    "enchanted library with floating books, magical light, fantasy illustration, highly detailed",
    "macro shot of dewdrop on leaf, morning light, bokeh background, ultra detailed, 8k",
    "snowy village at Christmas, warm lights, falling snow, cozy atmosphere, photorealistic",
    "aerial drone view of turquoise beach, palm trees, crystal water, highly detailed, 8k",
    "retro 80s synthwave sunset, palm silhouette, grid landscape, neon colors, cinematic"
];

const PROMPTS_IMG2IMG = [
    "replace background with dreamy bokeh garden, keep subject identical, soft natural light",
    "change clothing to elegant black suit, keep face and pose exactly, photorealistic",
    "add dramatic cinematic rim light, keep composition, ultra detailed, sharp focus",
    "transform to oil painting style, keep structure and colors, visible brush strokes, highly detailed",
    "make it autumn season with golden leaves and warm tones, keep original subject",
    "change to night scene with neon lights and reflections, keep foreground subject",
    "add light snowfall and winter atmosphere, keep subject, photorealistic, cozy",
    "convert to vintage film photo, grain, faded colors, 1970s aesthetic, keep composition",
    "replace background with luxury interior, keep person exactly, studio lighting, photorealistic",
    "turn into anime illustration, keep pose and expression, vibrant colors, highly detailed",
    "add stylish sunglasses and streetwear, keep face identity, photorealistic, sharp focus",
    "change hair to long blonde wavy hair, keep face, natural look, ultra detailed",
    "make background blurred studio backdrop, keep subject sharp, professional portrait, bokeh",
    "transform to soft watercolor painting, gentle washes, keep composition, artistic",
    "add futuristic cyberpunk armor, keep face, cinematic lighting, ultra detailed",
    "change to summer beach background, sunny, turquoise water, keep foreground subject intact",
    "make it high contrast black and white portrait, keep fine details, Ansel Adams style",
    "add blooming flower field around subject, keep subject centered, dreamy, photorealistic",
    "convert to 3D Pixar character style, keep likeness, cute, highly detailed, vibrant",
    "add golden hour sunset glow, warm light, keep shadows natural, photorealistic, cinematic"
];

function hasReferenceImage() {
    return !!(base64Ref && refImgObj && refImgObj.src && !refImgObj.src.endsWith(window.location.href));
}

function applyPromptSuggestion(promptText) {
    const ta = document.getElementById('prompt');
    if (ta) {
        ta.value = promptText;
        ta.focus();
        saveFormState();
        // small flash feedback
        ta.style.borderColor = 'var(--ps-cyan)';
        setTimeout(() => ta.style.borderColor = '', 600);
    }
}

function renderPromptSuggestions() {
    const container = document.getElementById('suggestionChips');
    const label = document.getElementById('suggestionLabel');
    if (!container) return;
    const useRef = hasReferenceImage();
    const list = useRef ? PROMPTS_IMG2IMG : PROMPTS_TXT2IMG;
    if (label) {
        label.innerText = useRef
            ? "💡 20 Vorschläge — mit Referenzbild (Inpaint / Edit)"
            : "💡 20 Vorschläge — Text-to-Image";
    }
    container.innerHTML = '';
    list.forEach((p, idx) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'ps-pill ps-suggestion-pill';
        btn.title = p;
        // short label: first ~48 chars + …
        const short = p.length > 52 ? p.slice(0, 52) + "…" : p;
        btn.innerText = `${idx + 1}. ${short}`;
        btn.onclick = () => applyPromptSuggestion(p);
        container.appendChild(btn);
    });
}

function saveFormState() {
    const lorasList = [];
    document.querySelectorAll('.lora-path-sel').forEach(sel => {
        const path = sel.value;
        const scaleEl = sel.closest('div').parentElement.querySelector('.lora-scale-val');
        const scale = scaleEl ? parseFloat(scaleEl.value) : 1.0;
        if (path) lorasList.push({ path, scale });
    });

    const panelsState = {};
    ['panelFlux', 'panelLoras', 'panelCanvasSizing', 'panelSeg'].forEach(id => {
        const el = document.getElementById(id);
        if (el) panelsState[id] = el.classList.contains('collapsed');
    });

    const state = {
        prompt: document.getElementById('prompt').value,
        negative_prompt: document.getElementById('negative_prompt').value,
        width: document.getElementById('width').value,
        height: document.getElementById('height').value,
        steps: document.getElementById('steps').value,
        cfg: document.getElementById('cfg').value,
        seed: document.getElementById('seed').value,
        panels: panelsState,
        loras: lorasList
    };
    localStorage.setItem('bonsai_flux_form_state', JSON.stringify(state));
}

function restoreFormState() {
    try {
        const raw = localStorage.getItem('bonsai_flux_form_state');
        if (!raw) return;
        const s = JSON.parse(raw);
        if (s.prompt) document.getElementById('prompt').value = s.prompt;
        if (s.negative_prompt) document.getElementById('negative_prompt').value = s.negative_prompt;
        if (s.width) document.getElementById('width').value = s.width;
        if (s.height) document.getElementById('height').value = s.height;
        if (s.steps) document.getElementById('steps').value = s.steps;
        if (s.cfg) document.getElementById('cfg').value = s.cfg;
        if (s.seed) document.getElementById('seed').value = s.seed;

        if (s.panels) {
            Object.keys(s.panels).forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.toggle('collapsed', !!s.panels[id]);
            });
        }

        if (Array.isArray(s.loras) && s.loras.length > 0) {
            const container = document.getElementById('loraContainer');
            if (container) container.innerHTML = '';
            s.loras.forEach(l => {
                addLoraRow(l.path, l.scale);
            });
        }
    } catch (_) {}
}

function togglePanel(panelId) {
    if (panelId === 'panelNegPrompt') {
        const el = document.getElementById('panelNegPrompt');
        const icon = document.getElementById('negToggleIcon');
        if (el) {
            const isHidden = el.style.display === 'none';
            el.style.display = isHidden ? 'block' : 'none';
            if (icon) icon.innerText = isHidden ? '▼' : '▶';
        }
        return;
    }
    const el = document.getElementById(panelId);
    if (!el) return;
    el.classList.toggle('collapsed');
    saveFormState();
}

async function loadLoras() {
    try {
        const res = await fetch('/v1/loras');
        if (res.ok) availableLoras = await res.json();
    } catch (_) {}
}

let viewMode = 'mix'; // 'original', 'mix', 'generated'
let showMaskOverlay = true;

function toggleMaskOverlayVisibility() {
    showMaskOverlay = !showMaskOverlay;
    const btn = document.getElementById('btnToggleMaskOverlay');
    if (btn) {
        btn.classList.toggle('active', showMaskOverlay);
        btn.style.opacity = showMaskOverlay ? '1.0' : '0.45';
        btn.innerText = showMaskOverlay ? '🎭 Mask (Red)' : '🚫 Mask Hidden';
    }
    renderLiveComposite(false);
}

function selectTool(tool) {
    activeTool = tool;
    document.getElementById('toolBrush').classList.toggle('active', tool === 'brush');
    document.getElementById('toolEraser').classList.toggle('active', tool === 'eraser');
    document.getElementById('toolWand').classList.toggle('active', tool === 'wand');
    document.getElementById('toolFill').classList.toggle('active', tool === 'fill');

    if (tool === 'fill') {
        fillEntireMask();
    }
}

function setViewMode(mode) {
    viewMode = mode;
    ['original', 'mix', 'generated'].forEach(m => {
        const btn = document.getElementById('vm' + m.charAt(0).toUpperCase() + m.slice(1));
        if (btn) btn.classList.toggle('active', m === mode);
    });
    renderLiveComposite(false);
}

function addLoraRow(defaultPath = '', defaultScale = 1.0) {
    const container = document.getElementById('loraContainer');
    const rowId = 'lora_' + Math.random().toString(36).substr(2, 9);
    const div = document.createElement('div');
    div.className = 'lora-row';
    div.id = rowId;

    let optionsHtml = '<option value="">-- Select LoRA --</option>';
    availableLoras.forEach(l => {
        const selected = l.path === defaultPath ? 'selected' : '';
        optionsHtml += `<option value="${l.path}" ${selected}>${l.name}</option>`;
    });

    div.innerHTML = `
        <div class="lora-header">
            <select class="lora-path-sel ps-select" onchange="saveFormState()" style="flex:1; margin-right:6px;">${optionsHtml}</select>
            <button type="button" class="ps-pill" onclick="document.getElementById('${rowId}').remove(); saveFormState();" style="color:var(--ps-danger);">✕</button>
        </div>
        <div class="lora-scale-row">
            <span style="color:var(--ps-text-dim); font-size:10px;">Scale: <strong class="lora-scale-lbl" style="color:var(--ps-cyan);">${defaultScale.toFixed(2)}</strong></span>
            <input type="range" class="lora-scale-val" min="0.0" max="2.0" step="0.05" value="${defaultScale}" oninput="this.parentElement.querySelector('.lora-scale-lbl').innerText = parseFloat(this.value).toFixed(2); saveFormState();" style="width:120px; accent-color:var(--ps-cyan);">
        </div>
    `;
    container.appendChild(div);
    saveFormState();
}

function initCanvas() {
    mainCanvas = document.getElementById('mainCanvas');
    mainCtx = mainCanvas.getContext('2d', { willReadFrequently: true });

    maskCanvas = document.createElement('canvas');
    maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });

    function getPos(e) {
        const rect = mainCanvas.getBoundingClientRect();
        const scaleX = mainCanvas.width / rect.width;
        const scaleY = mainCanvas.height / rect.height;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDraw(e) {
        const pos = getPos(e);

        if (activeTool === 'wand') {
            handleWandClick(pos.x, pos.y);
            return;
        }

        isDrawing = true;
        saveMaskUndoState();

        maskCtx.beginPath();
        maskCtx.moveTo(pos.x, pos.y);
        draw(e);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        const size = parseInt(document.getElementById('brushSize').value);

        if (activeTool === 'eraser') {
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(pos.x, pos.y, size / 2, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.restore();
        } else {
            maskCtx.lineWidth = size;
            maskCtx.lineCap = 'round';
            maskCtx.lineJoin = 'round';
            maskCtx.strokeStyle = '#ffffff';
            maskCtx.lineTo(pos.x, pos.y);
            maskCtx.stroke();
        }

        renderLiveComposite(true);
    }

    function stopDraw() {
        if (isDrawing) {
            isDrawing = false;
            maskCtx.closePath();
            renderLiveComposite(false);
        }
    }

    mainCanvas.addEventListener('mousedown', startDraw);
    mainCanvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDraw);

    mainCanvas.addEventListener('touchstart', startDraw, { passive: false });
    mainCanvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDraw);

    setSize(512, 512);
}

function saveMaskUndoState() {
    if (!maskCanvas || !maskCtx) return;
    maskUndoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
    if (maskUndoStack.length > MAX_UNDO) maskUndoStack.shift();
}

function undoMask() {
    if (maskUndoStack.length === 0) return;
    const lastState = maskUndoStack.pop();
    maskCtx.putImageData(lastState, 0, 0);
    renderLiveComposite(false);
}

function clearMask() {
    saveMaskUndoState();
    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (detectedSegments.length > 0) {
        detectedSegments.forEach(s => s.selected = false);
        renderSegmentChips();
    }
    renderLiveComposite(false);
}

function fillEntireMask() {
    saveMaskUndoState();
    maskCtx.fillStyle = '#ffffff';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    if (detectedSegments.length > 0) {
        detectedSegments.forEach(s => s.selected = true);
        renderSegmentChips();
    }
    renderLiveComposite(false);
}

function hasMaskDrawn() {
    if (!maskCanvas || !maskCtx) return false;
    const imgData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
    for (let i = 3; i < imgData.length; i += 4) {
        if (imgData[i] > 10) return true;
    }
    return false;
}

function setSize(w, h, btnEl = null) {
    document.getElementById('width').value = w;
    document.getElementById('height').value = h;
    document.getElementById('docDimsStatus').innerText = `${w} × ${h} px`;
    document.getElementById('statusRes').innerText = `${w}×${h}`;
    saveFormState();

    if (btnEl) {
        document.querySelectorAll('.ps-pills .ps-pill').forEach(p => p.classList.remove('active'));
        btnEl.classList.add('active');
    }

    mainCanvas.width = w;
    mainCanvas.height = h;
    maskCanvas.width = w;
    maskCanvas.height = h;

    renderLiveComposite(false);
}

function applyImageSizing() {
    if (!rawRefWidth || !rawRefHeight) return;
    const targetW = croppedGridWidth;
    const targetH = croppedGridHeight;

    document.getElementById('width').value = targetW;
    document.getElementById('height').value = targetH;
    document.getElementById('docDimsStatus').innerText = `${targetW} × ${targetH} px`;
    document.getElementById('statusRes').innerText = `${targetW}×${targetH}`;
    saveFormState();

    mainCanvas.width = targetW;
    mainCanvas.height = targetH;
    maskCanvas.width = targetW;
    maskCanvas.height = targetH;

    renderLiveComposite(false);
}

function handleImage(e) {
    const files = Array.from(e.target.files || []);
    if (!files || files.length === 0) return;

    // Load each selected file
    files.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(evt) {
            const dataUrl = evt.target.result;

            const tempImg = new Image();
            tempImg.onload = () => {
                const w = tempImg.naturalWidth || tempImg.width;
                const h = tempImg.naturalHeight || tempImg.height;
                const gridW = Math.min(2048, Math.max(256, Math.floor(w / 16) * 16));
                const gridH = Math.min(2048, Math.max(256, Math.floor(h / 16) * 16));

                // If first image (or only one), load it directly into Layer 0
                if (index === 0) {
                    base64Ref = dataUrl;
                    const dropText = document.getElementById('dropzoneText');
                    if (dropText) {
                        dropText.innerText = files.length > 1
                            ? `✅ Loaded ${files.length} Images (${file.name})`
                            : `✅ ${file.name}`;
                    }
                    refImgObj.onload = () => {
                        rawRefWidth = refImgObj.naturalWidth || refImgObj.width;
                        rawRefHeight = refImgObj.naturalHeight || refImgObj.height;
                        croppedGridWidth = gridW;
                        croppedGridHeight = gridH;

                        applyImageSizing();
                        renderPromptSuggestions();
                    };
                    refImgObj.src = dataUrl;
                    // immediate switch for UI feedback (before onload fires)
                    setTimeout(() => renderPromptSuggestions(), 80);
                }
            };
            tempImg.src = dataUrl;
        };
        reader.readAsDataURL(file);
    });
}

/**
 * Build a feathered mask canvas that can be used as alpha source.
 * Returns a canvas where white = generate (inpaint here), black = keep original.
 */
function _buildAlphaMask(w, h) {
    const feather = parseInt(document.getElementById('maskFeather').value);

    const fc = document.createElement('canvas');
    fc.width = w; fc.height = h;
    const fCtx = fc.getContext('2d');

    if (feather > 0) fCtx.filter = `blur(${feather}px)`;
    fCtx.drawImage(maskCanvas, 0, 0, w, h);
    return fc;
}

/**
 * Single Unified Canvas Rendering Engine:
 * Always displays the direct current image + live generation preview blended inside the mask during sampling.
 */
function renderLiveComposite(isDrawingBrush = false) {
    const w = mainCanvas.width;
    const h = mainCanvas.height;

    mainCtx.clearRect(0, 0, w, h);

    const hasRef = refImgObj && refImgObj.src && !refImgObj.src.endsWith(window.location.href);
    const hasGen = latestGeneratedImgObj && latestGeneratedImgObj.src && !latestGeneratedImgObj.src.endsWith(window.location.href);

    // 1. Draw current active image
    if (hasRef) {
        mainCtx.drawImage(refImgObj, 0, 0, w, h);
    } else {
        mainCtx.fillStyle = '#181818';
        mainCtx.fillRect(0, 0, w, h);
    }

    // 2. If currently generating (live preview step), composite strictly inside the mask
    if (hasGen && latestGeneratedImgObj.complete && latestGeneratedImgObj.naturalWidth > 0) {
        const genC = document.createElement('canvas');
        genC.width = w; genC.height = h;
        const gCtx = genC.getContext('2d');
        gCtx.drawImage(latestGeneratedImgObj, 0, 0, w, h);

        if (hasMaskDrawn()) {
            const alphaMask = _buildAlphaMask(w, h);
            gCtx.globalCompositeOperation = 'destination-in';
            gCtx.drawImage(alphaMask, 0, 0, w, h);
        }
        mainCtx.drawImage(genC, 0, 0, w, h);
    }

    // 3. Mask overlay preview (red) — stays visible while drawing AND during generation
    if (hasMaskDrawn() && !showSegOverlay) {
        const opacityEl = document.getElementById('maskOpacity');
        const opacity = opacityEl ? parseInt(opacityEl.value) / 100.0 : 1.0;

        const alphaMask = _buildAlphaMask(w, h);
        const redC = document.createElement('canvas');
        redC.width = w; redC.height = h;
        const rCtx = redC.getContext('2d');
        rCtx.fillStyle = `rgba(239, 68, 68, ${opacity})`;
        rCtx.fillRect(0, 0, w, h);
        rCtx.globalCompositeOperation = 'destination-in';
        rCtx.drawImage(alphaMask, 0, 0, w, h);
        mainCtx.drawImage(redC, 0, 0, w, h);
    }
}

/**
 * Capture current base image for the generation payload.
 */
function getCleanCompositeDataURL() {
    const w = mainCanvas.width;
    const h = mainCanvas.height;
    const off = document.createElement('canvas');
    off.width = w; off.height = h;
    const ctx = off.getContext('2d');

    const hasRef = refImgObj && refImgObj.src && !refImgObj.src.endsWith(window.location.href);
    if (hasRef) {
        ctx.drawImage(refImgObj, 0, 0, w, h);
    }

    return off.toDataURL('image/png');
}

async function harmonizeComposite() {
    const compositeB64 = getCleanCompositeDataURL();
    base64Ref = compositeB64;
    refImgObj.src = compositeB64;
    await generate();
}

async function generate() {
    const prompt = document.getElementById('prompt').value.trim();
    if (!prompt) {
        document.getElementById('statusDuration').innerText = "⚠️ Please enter a positive prompt!";
        return;
    }
    const negPrompt = document.getElementById('negative_prompt').value.trim();

    const width = parseInt(document.getElementById('width').value);
    const height = parseInt(document.getElementById('height').value);
    const steps = parseInt(document.getElementById('steps').value);
    const cfg = parseFloat(document.getElementById('cfg').value);
    let seed = parseInt(document.getElementById('seed').value);
    if (seed === -1) seed = Math.floor(Math.random() * 2147483647);

    saveFormState();

    let currentGroundTruthRef = base64Ref;
    if (refImgObj.src || latestGeneratedImgObj.src) {
        currentGroundTruthRef = getCleanCompositeDataURL();
        base64Ref = currentGroundTruthRef;
        refImgObj.src = currentGroundTruthRef;
    }

    const lorasList = [];
    document.querySelectorAll('.lora-path-sel').forEach(sel => {
        const path = sel.value;
        const scaleEl = sel.closest('div').parentElement.querySelector('.lora-scale-val');
        const scale = scaleEl ? parseFloat(scaleEl.value) : 1.0;
        if (path) lorasList.push({ path: path, scale: scale });
    });

    const btn = document.getElementById('genBtn');
    const hud = document.getElementById('loadingHud');
    const hudStage = document.getElementById('hudStage');
    const hudEta = document.getElementById('hudEta');
    const hudBar = document.getElementById('hudBar');
    const hudPct = document.getElementById('hudPct');

    btn.disabled = true;
    hud.style.display = 'flex';
    hudBar.style.width = '0%';
    hudStage.innerText = "Initializing Pipeline...";
    hudEta.innerText = "ETA: Calculating...";
    if (hudPct) hudPct.innerText = "0%";

    let maskPayload = null;
    if (hasMaskDrawn() && maskCanvas) {
        maskPayload = maskCanvas.toDataURL("image/png");
    }

    const startTime = performance.now();
    if (progressPollInterval) clearInterval(progressPollInterval);

    latestGeneratedImgObj.onload = null;

    let pollErrorCount = 0;
    progressPollInterval = setInterval(async () => {
        try {
            const progRes = await fetch('/v1/progress', { cache: 'no-store' });
            if (progRes.ok) {
                pollErrorCount = 0;
                const pData = await progRes.json();
                if (pData.active) {
                    hudStage.innerText = `${pData.stage} (${pData.rate_str})`;
                    hudEta.innerText = pData.eta_s > 0 ? `ETA: ~${pData.eta_s}s` : 'Sampling...';
                    hudBar.style.width = `${Math.max(5, pData.pct)}%`;
                    if (hudPct) hudPct.innerText = `${Math.round(pData.pct)}%`;

                    if (pData.preview_b64) {
                        latestGeneratedImgObj.src = "data:image/jpeg;base64," + pData.preview_b64;
                        renderLiveComposite(false);
                    }
                }
            } else {
                pollErrorCount++;
            }
        } catch (_) {
            pollErrorCount++;
            if (pollErrorCount > 15) {
                hudStage.innerText = "Reconnecting to server...";
            }
        }
    }, 250);

    const payload = {
        prompt, negative_prompt: negPrompt,
        width, height, steps, cfg_scale: cfg, seed,
        init_image: currentGroundTruthRef,
        mask_image: maskPayload,
        mask_blur: parseInt(document.getElementById('maskFeather').value) || 8,
        invert_mask: false,
        loras: lorasList
    };

    try {
        // Robust fetch with retry on temporary network glitch
        let res = null;
        let lastErr = null;
        for (let attempt = 1; attempt <= 3; attempt++) {
            try {
                res = await fetch('/v1/images/generations', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res) break;
            } catch (fetchErr) {
                lastErr = fetchErr;
                console.warn(`Fetch generation attempt ${attempt} failed:`, fetchErr);
                if (attempt < 3) {
                    hudStage.innerText = `Waiting for server (Retry ${attempt}/3)...`;
                    await new Promise(r => setTimeout(r, 1500));
                }
            }
        }

        if (!res && lastErr) {
            throw lastErr;
        }

        if (!res.ok) {
            let errorMsg = `Server error ${res.status}`;
            try {
                const errJson = await res.json();
                errorMsg = errJson.detail || errorMsg;
            } catch (_) {}
            throw new Error(errorMsg);
        }

        const data = await res.json();
        if (!data || !data.data || !data.data[0] || !data.data[0].b64_json) {
            throw new Error("Invalid response received from server");
        }

        const b64 = data.data[0].b64_json;
        const dur = ((performance.now() - startTime) / 1000).toFixed(2);
        const finalSeed = data.data[0].seed !== undefined ? data.data[0].seed : seed;

        await new Promise(resolve => {
            const tempImg = new Image();
            tempImg.onload = () => {
                // Update ground truth reference with the new generated composite image
                base64Ref = "data:image/png;base64," + b64;
                refImgObj.src = base64Ref;
                latestGeneratedImgObj.src = "";
                latestGeneratedB64 = base64Ref;

                document.getElementById('statusDuration').innerText =
                    `Rendered in ${dur}s (${steps} steps, Seed: ${finalSeed})`;
                renderLiveComposite(false);
                renderPromptSuggestions();
                resolve();
            };
            tempImg.onerror = () => {
                resolve();
            };
            tempImg.src = "data:image/png;base64," + b64;
        });

    } catch (err) {
        console.error("Generation Error:", err);
        document.getElementById('statusDuration').innerText = "⚠️ Generation Error: " + err.message;
    } finally {
        if (progressPollInterval) clearInterval(progressPollInterval);
        progressPollInterval = null;
        btn.disabled = false;
        hud.style.display = 'none';
    }
}

// Global Event Listeners
window.addEventListener('DOMContentLoaded', async () => {
    await loadLoras();
    restoreFormState();
    initCanvas();
    renderPromptSuggestions();

    ['prompt', 'negative_prompt', 'width', 'height', 'steps', 'cfg', 'seed'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', saveFormState);
    });
});
