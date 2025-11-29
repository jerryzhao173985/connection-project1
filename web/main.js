/**
 * Hidden Connections - Semantic Nebula
 *
 * A cinematic visualization where:
 * - Each participant is a softly glowing star
 * - Clusters form colored nebula clouds in the background
 * - Nearest neighbors are connected by delicate constellation lines
 * - Everything slowly moves with breathing and pulsing animations
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const CONFIG = {
    // Colors for clusters (nebula palette)
    colors: [
        { r: 78, g: 205, b: 196 },   // teal
        { r: 255, g: 107, b: 107 },  // coral
        { r: 201, g: 177, b: 255 },  // lavender
        { r: 255, g: 230, b: 109 },  // warm yellow
        { r: 149, g: 225, b: 211 },  // mint
        { r: 243, g: 129, b: 129 },  // salmon
        { r: 116, g: 185, b: 255 },  // sky blue
        { r: 255, g: 234, b: 167 },  // light gold
        { r: 253, g: 121, b: 168 },  // pink
        { r: 162, g: 155, b: 254 },  // purple
    ],

    // Animation speeds
    breatheSpeed: 0.0008,      // How fast stars breathe
    pulseSpeed: 0.002,         // How fast link pulses travel
    nebulaSpeed: 0.0003,       // How fast nebula clouds flow

    // Star rendering
    starBaseRadius: 3,
    starGlowMultiplier: 4,

    // Connections
    neighborCount: 4,          // How many neighbors to connect per star
    maxConnectionDist: 0.25,   // Max distance for connections (in normalized coords)

    // Hover effects
    hoverBrightness: 2.5,
    neighborBrightness: 1.8,
    dimOpacity: 0.15,

    // Touch
    touchThreshold: 50,
    touchHoldTime: 4000,
};

// =============================================================================
// STATE
// =============================================================================

let points = [];
let connections = [];  // Pre-computed neighbor connections
let hoveredPoint = null;
let hoveredNeighbors = [];
let selfPoint = null;

// Canvases
let canvas, ctx;
let nebulaCanvas, nebulaCtx;
let dpr = 1;
let width, height;
const padding = 60;

// Animation
let time = 0;
let lastTime = 0;
let animationFrame = null;

// Touch state
let touchActive = false;
let touchTimeout = null;
let lastTouchTime = 0;

// Nebula texture
let nebulaImageData = null;
let nebulaTime = 0;

// Self-submission
let embeddingPipeline = null;
let isLoadingModel = false;

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    setupCanvases();
    await loadData();
    computeConnections();
    setupEventListeners();
    startAnimation();
}

function setupCanvases() {
    dpr = window.devicePixelRatio || 1;
    width = window.innerWidth;
    height = window.innerHeight;

    // Main canvas for stars
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(dpr, dpr);

    // Nebula background canvas
    nebulaCanvas = document.getElementById('nebula-canvas');
    nebulaCtx = nebulaCanvas.getContext('2d');
    nebulaCanvas.width = width * dpr;
    nebulaCanvas.height = height * dpr;
    nebulaCanvas.style.width = width + 'px';
    nebulaCanvas.style.height = height + 'px';
    nebulaCtx.scale(dpr, dpr);
}

async function loadData() {
    try {
        const response = await fetch('points.json');
        points = await response.json();

        // Update stats
        document.getElementById('point-count').textContent = points.length;
        const clusters = new Set(points.map(p => p.cluster));
        document.getElementById('cluster-count').textContent = clusters.size;

        console.log(`Loaded ${points.length} souls in ${clusters.size} clusters`);
    } catch (err) {
        console.error('Failed to load data:', err);
    }
}

function computeConnections() {
    // For each point, find its nearest neighbors within the same cluster
    connections = [];

    for (const point of points) {
        // Get points in same cluster
        const sameCluster = points.filter(p =>
            p.id !== point.id && p.cluster === point.cluster
        );

        // Calculate distances
        const withDist = sameCluster.map(p => ({
            from: point,
            to: p,
            dist: Math.hypot(point.x - p.x, point.y - p.y)
        }));

        // Sort by distance and take nearest
        withDist.sort((a, b) => a.dist - b.dist);
        const nearest = withDist.slice(0, CONFIG.neighborCount);

        // Add connections (avoid duplicates by only adding if from.id < to.id)
        for (const conn of nearest) {
            if (conn.dist < CONFIG.maxConnectionDist) {
                if (conn.from.id < conn.to.id) {
                    connections.push(conn);
                }
            }
        }
    }

    console.log(`Computed ${connections.length} constellation links`);
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

function setupEventListeners() {
    // Resize
    window.addEventListener('resize', handleResize);

    // Mouse
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    // Touch
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd, { passive: true });
    canvas.addEventListener('touchcancel', handleTouchEnd, { passive: true });

    // Keyboard
    document.addEventListener('keydown', handleKeyboard);

    // Buttons
    document.getElementById('btn-add-self').addEventListener('click', openSubmitModal);
    document.getElementById('btn-help').addEventListener('click', openHelpModal);
    document.getElementById('modal-close').addEventListener('click', closeSubmitModal);
    document.getElementById('cancel-btn').addEventListener('click', closeSubmitModal);
    document.getElementById('help-close').addEventListener('click', closeHelpModal);

    // Modal backgrounds
    document.getElementById('submit-modal').addEventListener('click', (e) => {
        if (e.target.id === 'submit-modal') closeSubmitModal();
    });
    document.getElementById('help-modal').addEventListener('click', (e) => {
        if (e.target.id === 'help-modal') closeHelpModal();
    });

    // Form
    document.getElementById('self-form').addEventListener('submit', handleSelfSubmit);
}

function handleResize() {
    setupCanvases();
    computeConnections();
}

function handleKeyboard(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
        case '?':
            toggleHelpModal();
            break;
        case ' ':
            e.preventDefault();
            openSubmitModal();
            break;
        case 'Escape':
            closeSubmitModal();
            closeHelpModal();
            break;
    }
}

// =============================================================================
// MOUSE/TOUCH HANDLERS
// =============================================================================

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    findPointAt(x, y, 25);
}

function handleMouseLeave() {
    setHoveredPoint(null);
}

function handleTouchStart(e) {
    const pos = getTouchPos(e);
    if (!pos) return;

    if (touchTimeout) {
        clearTimeout(touchTimeout);
        touchTimeout = null;
    }

    touchActive = true;
    lastTouchTime = Date.now();

    const found = findPointAt(pos.x, pos.y, CONFIG.touchThreshold);
    if (found) {
        e.preventDefault();
    }
}

function handleTouchMove(e) {
    if (!touchActive) return;

    const pos = getTouchPos(e);
    if (!pos) return;

    lastTouchTime = Date.now();
    const found = findPointAt(pos.x, pos.y, CONFIG.touchThreshold);
    if (found) {
        e.preventDefault();
    }
}

function handleTouchEnd() {
    touchActive = false;

    touchTimeout = setTimeout(() => {
        if (Date.now() - lastTouchTime >= CONFIG.touchHoldTime - 100) {
            setHoveredPoint(null);
        }
    }, CONFIG.touchHoldTime);
}

function getTouchPos(e) {
    const touch = e.touches[0] || e.changedTouches[0];
    if (!touch) return null;
    const rect = canvas.getBoundingClientRect();
    return {
        x: touch.clientX - rect.left,
        y: touch.clientY - rect.top
    };
}

function findPointAt(mouseX, mouseY, threshold) {
    let nearest = null;
    let minDist = Infinity;

    for (const point of points) {
        const [px, py] = dataToScreen(point.x, point.y);
        const dist = Math.hypot(mouseX - px, mouseY - py);

        if (dist < threshold && dist < minDist) {
            minDist = dist;
            nearest = point;
        }
    }

    setHoveredPoint(nearest);
    return nearest;
}

function setHoveredPoint(point) {
    if (point === hoveredPoint) return;

    hoveredPoint = point;

    if (hoveredPoint) {
        // Find neighbors for this point
        hoveredNeighbors = findNeighbors(hoveredPoint);
        updatePanel(hoveredPoint);
        showPanel();
    } else {
        hoveredNeighbors = [];
        hidePanel();
    }
}

function findNeighbors(point) {
    const neighbors = new Set();

    for (const conn of connections) {
        if (conn.from.id === point.id) {
            neighbors.add(conn.to);
        } else if (conn.to.id === point.id) {
            neighbors.add(conn.from);
        }
    }

    return Array.from(neighbors);
}

// =============================================================================
// COORDINATE TRANSFORMATION
// =============================================================================

function dataToScreen(x, y) {
    const effectiveWidth = width - padding * 2;
    const effectiveHeight = height - padding * 2;

    const screenX = padding + ((x + 1) / 2) * effectiveWidth;
    const screenY = padding + ((1 - y) / 2) * effectiveHeight;

    return [screenX, screenY];
}

// =============================================================================
// ANIMATION LOOP
// =============================================================================

function startAnimation() {
    lastTime = performance.now();
    animate();
}

function animate(currentTime = 0) {
    const deltaTime = currentTime - lastTime;
    lastTime = currentTime;
    time += deltaTime;
    nebulaTime += deltaTime;

    // Render nebula background (less frequently for performance)
    if (Math.floor(nebulaTime / 50) !== Math.floor((nebulaTime - deltaTime) / 50)) {
        renderNebula();
    }

    // Render stars and connections
    renderMain();

    animationFrame = requestAnimationFrame(animate);
}

// =============================================================================
// NEBULA RENDERING
// =============================================================================

function renderNebula() {
    // Clear
    nebulaCtx.fillStyle = '#050508';
    nebulaCtx.fillRect(0, 0, width, height);

    // Group points by cluster
    const clusters = new Map();
    for (const point of points) {
        if (!clusters.has(point.cluster)) {
            clusters.set(point.cluster, []);
        }
        clusters.get(point.cluster).push(point);
    }

    // Draw nebula cloud for each cluster
    for (const [clusterId, clusterPoints] of clusters) {
        if (clusterPoints.length < 2) continue;

        const color = CONFIG.colors[clusterId % CONFIG.colors.length];

        // Calculate cluster center and spread
        let cx = 0, cy = 0;
        for (const p of clusterPoints) {
            cx += p.x;
            cy += p.y;
        }
        cx /= clusterPoints.length;
        cy /= clusterPoints.length;

        // Calculate spread
        let maxDist = 0;
        for (const p of clusterPoints) {
            const d = Math.hypot(p.x - cx, p.y - cy);
            if (d > maxDist) maxDist = d;
        }

        const [screenCx, screenCy] = dataToScreen(cx, cy);
        const screenRadius = maxDist * Math.min(width, height) * 0.4;

        // Add noise-based offset for flowing effect
        const noiseOffset = simpleNoise(clusterId * 100, nebulaTime * CONFIG.nebulaSpeed);
        const offsetX = noiseOffset * 20;
        const offsetY = simpleNoise(clusterId * 200, nebulaTime * CONFIG.nebulaSpeed) * 20;

        // Draw soft gradient cloud
        const gradient = nebulaCtx.createRadialGradient(
            screenCx + offsetX, screenCy + offsetY, 0,
            screenCx + offsetX, screenCy + offsetY, screenRadius + 50
        );

        const alpha = 0.08 + 0.02 * Math.sin(nebulaTime * 0.0005 + clusterId);
        gradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 1.5})`);
        gradient.addColorStop(0.5, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

        nebulaCtx.fillStyle = gradient;
        nebulaCtx.beginPath();
        nebulaCtx.arc(screenCx + offsetX, screenCy + offsetY, screenRadius + 50, 0, Math.PI * 2);
        nebulaCtx.fill();

        // Draw secondary smaller clouds at some cluster points
        for (let i = 0; i < Math.min(3, clusterPoints.length); i++) {
            const p = clusterPoints[i];
            const [px, py] = dataToScreen(p.x, p.y);

            const smallGradient = nebulaCtx.createRadialGradient(px, py, 0, px, py, 40);
            smallGradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha * 0.5})`);
            smallGradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

            nebulaCtx.fillStyle = smallGradient;
            nebulaCtx.beginPath();
            nebulaCtx.arc(px, py, 40, 0, Math.PI * 2);
            nebulaCtx.fill();
        }
    }
}

// Simple noise function for nebula movement
function simpleNoise(seed, t) {
    return Math.sin(seed + t) * Math.cos(seed * 0.7 + t * 1.3) +
           Math.sin(seed * 1.5 + t * 0.8) * 0.5;
}

// =============================================================================
// MAIN RENDERING (Stars + Connections)
// =============================================================================

function renderMain() {
    // Clear with transparent
    ctx.clearRect(0, 0, width, height);

    // Draw connections first (behind stars)
    renderConnections();

    // Draw stars
    renderStars();

    // Update your marker position if exists
    if (selfPoint) {
        updateYourMarker();
    }
}

function renderConnections() {
    for (const conn of connections) {
        const [x1, y1] = dataToScreen(conn.from.x, conn.from.y);
        const [x2, y2] = dataToScreen(conn.to.x, conn.to.y);

        // Check if this connection involves hovered point
        const isActive = hoveredPoint && (
            conn.from.id === hoveredPoint.id ||
            conn.to.id === hoveredPoint.id
        );

        // Check if both ends are neighbors of hovered
        const bothNeighbors = hoveredPoint &&
            hoveredNeighbors.some(n => n.id === conn.from.id) &&
            hoveredNeighbors.some(n => n.id === conn.to.id);

        // Get cluster color
        const color = CONFIG.colors[conn.from.cluster % CONFIG.colors.length];

        // Calculate pulse effect
        const pulsePhase = (time * CONFIG.pulseSpeed + conn.from.id * 0.1) % 1;

        // Line properties based on state
        let opacity, lineWidth;

        if (isActive) {
            // This connection goes to/from hovered star
            opacity = 0.6 + 0.3 * Math.sin(pulsePhase * Math.PI * 2);
            lineWidth = 1.5 + 0.5 * Math.sin(pulsePhase * Math.PI * 2);
        } else if (bothNeighbors) {
            // Between two neighbors
            opacity = 0.25;
            lineWidth = 0.8;
        } else if (hoveredPoint) {
            // Something is hovered but not this connection
            opacity = 0.03;
            lineWidth = 0.5;
        } else {
            // Default: subtle pulsing
            opacity = 0.08 + 0.04 * Math.sin(pulsePhase * Math.PI * 2);
            lineWidth = 0.5;
        }

        // Draw line
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${opacity})`;
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        // Draw pulse traveling along the line for active connections
        if (isActive) {
            const pulseX = x1 + (x2 - x1) * pulsePhase;
            const pulseY = y1 + (y2 - y1) * pulsePhase;

            ctx.beginPath();
            ctx.arc(pulseX, pulseY, 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.8)`;
            ctx.fill();
        }
    }
}

function renderStars() {
    // Sort: hovered on top, then neighbors, then rest
    const sorted = [...points].sort((a, b) => {
        const aIsHovered = a === hoveredPoint;
        const bIsHovered = b === hoveredPoint;
        const aIsNeighbor = hoveredNeighbors.includes(a);
        const bIsNeighbor = hoveredNeighbors.includes(b);
        const aIsSelf = a === selfPoint;
        const bIsSelf = b === selfPoint;

        if (aIsHovered) return 1;
        if (bIsHovered) return -1;
        if (aIsSelf) return 1;
        if (bIsSelf) return -1;
        if (aIsNeighbor && !bIsNeighbor) return 1;
        if (!aIsNeighbor && bIsNeighbor) return -1;
        return 0;
    });

    for (const point of sorted) {
        drawStar(point);
    }
}

function drawStar(point) {
    const [x, y] = dataToScreen(point.x, point.y);
    const color = CONFIG.colors[point.cluster % CONFIG.colors.length];

    // Determine state
    const isHovered = point === hoveredPoint;
    const isNeighbor = hoveredNeighbors.includes(point);
    const isSelf = point === selfPoint;

    // Breathing animation
    const breathe = Math.sin(time * CONFIG.breatheSpeed + point.x * 10 + point.y * 10);

    // Calculate radius with breathing
    let baseRadius = CONFIG.starBaseRadius;
    if (isSelf) baseRadius = 5;
    else if (isHovered) baseRadius = 6;
    else if (isNeighbor) baseRadius = 4;

    const radius = baseRadius + breathe * 0.5;

    // Calculate brightness
    let brightness = 1;
    let opacity = 1;

    if (isHovered) {
        brightness = CONFIG.hoverBrightness;
    } else if (isNeighbor) {
        brightness = CONFIG.neighborBrightness;
    } else if (hoveredPoint) {
        opacity = CONFIG.dimOpacity;
    }

    // Glow radius
    const glowRadius = radius * CONFIG.starGlowMultiplier * brightness;

    // Draw outer glow
    const glowGradient = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
    const glowAlpha = 0.3 * brightness * opacity;
    glowGradient.addColorStop(0, `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha})`);
    glowGradient.addColorStop(0.4, `rgba(${color.r}, ${color.g}, ${color.b}, ${glowAlpha * 0.3})`);
    glowGradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, 0)`);

    ctx.beginPath();
    ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
    ctx.fillStyle = glowGradient;
    ctx.fill();

    // Draw core
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);

    // Core gradient for 3D effect
    const coreGradient = ctx.createRadialGradient(
        x - radius * 0.3, y - radius * 0.3, 0,
        x, y, radius
    );

    const coreAlpha = opacity;
    coreGradient.addColorStop(0, `rgba(255, 255, 255, ${coreAlpha})`);
    coreGradient.addColorStop(0.3, `rgba(${Math.min(255, color.r + 50)}, ${Math.min(255, color.g + 50)}, ${Math.min(255, color.b + 50)}, ${coreAlpha})`);
    coreGradient.addColorStop(1, `rgba(${color.r}, ${color.g}, ${color.b}, ${coreAlpha})`);

    ctx.fillStyle = coreGradient;
    ctx.fill();

    // Highlight ring for hovered/self
    if (isHovered || isSelf) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.8 * opacity})`;
        ctx.lineWidth = isSelf ? 2 : 1.5;
        ctx.stroke();

        // Outer pulsing ring
        const pulseRadius = radius + 8 + breathe * 2;
        ctx.beginPath();
        ctx.arc(x, y, pulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.3 * opacity})`;
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Neighbor highlight ring
    if (isNeighbor && !isHovered) {
        ctx.beginPath();
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`;
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// =============================================================================
// PANEL UPDATES
// =============================================================================

function showPanel() {
    document.getElementById('info-panel').classList.remove('hidden');
}

function hidePanel() {
    document.getElementById('info-panel').classList.add('hidden');
}

function updatePanel(point) {
    // Nickname
    document.getElementById('panel-nickname').textContent = point.nickname || 'anonymous';

    // Cluster badge with color
    const color = CONFIG.colors[point.cluster % CONFIG.colors.length];
    const clusterEl = document.getElementById('panel-cluster');
    clusterEl.textContent = `Cluster ${point.cluster}`;
    clusterEl.style.background = `rgba(${color.r}, ${color.g}, ${color.b}, 0.3)`;
    clusterEl.style.borderColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.5)`;

    // Questions
    const bodyEl = document.getElementById('panel-body');
    const questions = parseQuestions(point.text);

    bodyEl.innerHTML = questions.map(q => `
        <div class="question-block">
            <div class="question-label">${q.label}</div>
            <div class="question-text">${q.text}</div>
        </div>
    `).join('');

    // Meta info
    const metaEl = document.getElementById('panel-meta');
    const decisionStyle = point.decision_style || point.decisionStyle || '';
    const socialEnergy = point.social_energy || point.socialEnergy || '';
    const region = point.region || '';

    const metaItems = [];
    if (decisionStyle) metaItems.push(`<span class="meta-item">Decision: <span>${decisionStyle}</span></span>`);
    if (socialEnergy) metaItems.push(`<span class="meta-item">Energy: <span>${socialEnergy}</span></span>`);
    if (region) metaItems.push(`<span class="meta-item">Region: <span>${region}</span></span>`);

    metaEl.innerHTML = metaItems.join('');
}

function parseQuestions(text) {
    const questions = [];
    const patterns = [
        { regex: /Q1 \(([^)]+)\):\s*/, fallback: 'Safe Place' },
        { regex: /Q2 \(([^)]+)\):\s*/, fallback: 'Handling Stress' },
        { regex: /Q3 \(([^)]+)\):\s*/, fallback: 'Feeling Understood' },
        { regex: /Q4 \(([^)]+)\):\s*/, fallback: 'Free Day' },
        { regex: /Q5 \(([^)]+)\):\s*/, fallback: 'One Word' },
    ];

    const lines = text.split('\n');

    for (const line of lines) {
        for (const { regex, fallback } of patterns) {
            const match = line.match(regex);
            if (match) {
                const label = match[1] || fallback;
                const content = line.replace(regex, '').trim();
                if (content) {
                    questions.push({ label, text: content });
                }
                break;
            }
        }
    }

    // Fallback
    if (questions.length === 0) {
        const parts = text.split('\n').filter(l => l.trim());
        parts.forEach((part, i) => {
            questions.push({
                label: `Response ${i + 1}`,
                text: part.trim()
            });
        });
    }

    return questions;
}

// =============================================================================
// YOUR MARKER
// =============================================================================

function updateYourMarker() {
    const marker = document.getElementById('your-marker');
    const [x, y] = dataToScreen(selfPoint.x, selfPoint.y);

    marker.style.left = `${x}px`;
    marker.style.top = `${y - 20}px`;
}

// =============================================================================
// MODALS
// =============================================================================

function openSubmitModal() {
    document.getElementById('submit-modal').classList.remove('hidden');
}

function closeSubmitModal() {
    document.getElementById('submit-modal').classList.add('hidden');
}

function openHelpModal() {
    document.getElementById('help-modal').classList.remove('hidden');
}

function closeHelpModal() {
    document.getElementById('help-modal').classList.add('hidden');
}

function toggleHelpModal() {
    document.getElementById('help-modal').classList.toggle('hidden');
}

// =============================================================================
// SELF-SUBMISSION
// =============================================================================

async function handleSelfSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    // Show processing
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoading = submitBtn.querySelector('.btn-loading');
    const statusEl = document.getElementById('processing-status');
    const statusText = document.getElementById('status-text');

    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    submitBtn.disabled = true;
    statusEl.classList.remove('hidden');
    form.style.display = 'none';

    try {
        // Build text
        const userText = [
            `Q1 (safe place): ${formData.get('q1_safe_place')}`,
            `Q2 (stress): ${formData.get('q2_stress')}`,
            `Q3 (understood): ${formData.get('q3_understood')}`,
            `Q4 (free day): ${formData.get('q4_free_day')}`,
            `Q5 (one word): ${formData.get('q5_one_word')}`,
        ].join('\n');

        // Load model
        statusText.textContent = 'Loading embedding model...';
        await loadEmbeddingModel(progress => {
            if (progress.status === 'downloading') {
                const pct = Math.round((progress.loaded / progress.total) * 100);
                statusText.textContent = `Downloading model... ${pct}%`;
            }
        });

        // Generate embedding
        statusText.textContent = 'Generating your embedding...';
        const userEmbedding = await generateEmbedding(userText);

        // Find position
        statusText.textContent = 'Finding your position...';
        const position = await findPosition(userEmbedding);

        // Find cluster
        statusText.textContent = 'Determining your cluster...';
        const cluster = findNearestCluster(position);

        // Create self point
        selfPoint = {
            id: 'self',
            text: userText,
            x: position.x,
            y: position.y,
            cluster: cluster,
            nickname: formData.get('nickname') || 'You',
            isSelf: true
        };

        // Add to points
        points.push(selfPoint);

        // Recompute connections
        computeConnections();

        // Update stats
        document.getElementById('point-count').textContent = points.length;

        // Show marker
        document.getElementById('your-marker').classList.remove('hidden');
        updateYourMarker();

        // Close modal
        closeSubmitModal();

        // Highlight self
        setTimeout(() => {
            setHoveredPoint(selfPoint);
        }, 300);

    } catch (err) {
        console.error('Self-submission error:', err);
        statusText.textContent = `Error: ${err.message}`;
    } finally {
        // Reset form UI after delay
        setTimeout(() => {
            btnText.classList.remove('hidden');
            btnLoading.classList.add('hidden');
            submitBtn.disabled = false;
            statusEl.classList.add('hidden');
            form.style.display = '';
        }, 2000);
    }
}

async function loadEmbeddingModel(progressCallback) {
    if (embeddingPipeline) return;
    if (isLoadingModel) {
        while (isLoadingModel) {
            await new Promise(r => setTimeout(r, 100));
        }
        return;
    }

    isLoadingModel = true;
    try {
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
        embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: true,
            progress_callback: progressCallback
        });
    } finally {
        isLoadingModel = false;
    }
}

async function generateEmbedding(text) {
    const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

async function findPosition(userEmbedding) {
    // Sample points for comparison
    const sampleSize = Math.min(points.length, 30);
    const sampled = points.length <= sampleSize ? points :
        [...points].sort(() => Math.random() - 0.5).slice(0, sampleSize);

    // Generate embeddings for samples
    const embeddings = [];
    for (const point of sampled) {
        const emb = await generateEmbedding(point.text);
        embeddings.push({ point, embedding: emb });
    }

    // Calculate similarities
    const similarities = embeddings.map(({ point, embedding }) => ({
        point,
        sim: cosineSimilarity(userEmbedding, embedding)
    }));

    similarities.sort((a, b) => b.sim - a.sim);

    // Weighted average of top k
    const k = Math.min(5, similarities.length);
    let totalWeight = 0;
    let x = 0, y = 0;

    for (let i = 0; i < k; i++) {
        const { point, sim } = similarities[i];
        const weight = Math.pow(Math.max(0, sim), 2);
        x += point.x * weight;
        y += point.y * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0) {
        x /= totalWeight;
        y /= totalWeight;
    }

    // Add jitter
    x += (Math.random() - 0.5) * 0.02;
    y += (Math.random() - 0.5) * 0.02;

    return { x, y };
}

function cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function findNearestCluster(position) {
    let nearest = 0;
    let minDist = Infinity;

    for (const point of points) {
        if (point.isSelf) continue;
        const dist = Math.hypot(position.x - point.x, position.y - point.y);
        if (dist < minDist) {
            minDist = dist;
            nearest = point.cluster;
        }
    }

    return nearest;
}

// =============================================================================
// START
// =============================================================================

init();
