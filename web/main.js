/**
 * Hidden Connections - Project 4 (Maximum Features)
 * Interactive Projective Semantic Map
 *
 * Features:
 * - Multiple viewing modes (cluster, energy, decision, region)
 * - Smooth color transitions on mode switch
 * - Nearest-neighbor highlighting on hover
 * - Touch support for tablets
 * - Keyboard shortcuts
 */

// =============================================================================
// COLOR PALETTES (from PRD)
// =============================================================================

const CLUSTER_COLORS = [
    '#4ECDC4', // teal
    '#FF6B6B', // coral
    '#C9B1FF', // lavender
    '#FFE66D', // warm yellow
    '#95E1D3', // mint
    '#F38181', // salmon
    '#74B9FF', // sky blue
    '#FFEAA7', // light gold
    '#DFE6E9', // silver
    '#FD79A8', // pink
];

const ENERGY_COLORS = {
    'Energised': '#FFD93D',
    'Drained': '#6C5CE7',
    'Depends': '#A8E6CF',
};

const DECISION_COLORS = {
    'Mostly rational': '#74B9FF',
    'Mostly emotional': '#FD79A8',
    'Depends': '#FFEAA7',
};

const REGION_COLORS = {
    'North America': '#FF6B6B',
    'Europe': '#4ECDC4',
    'East Asia': '#C9B1FF',
    'South Asia': '#FFE66D',
    'Southeast Asia': '#95E1D3',
    'Oceania': '#74B9FF',
    'South America': '#F38181',
    'Africa': '#FD79A8',
    'Middle East': '#FFEAA7',
};

const MODE_CONFIG = {
    cluster: {
        title: 'Model Clusters',
        subtitle: 'ML-detected semantic groupings',
        getColor: (p) => CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length],
        getKey: (p) => p.cluster,
        colors: CLUSTER_COLORS,
        formatLabel: (key) => `Cluster ${key}`,
    },
    energy: {
        title: 'Social Energy',
        subtitle: 'Self-reported energy from socializing',
        getColor: (p) => ENERGY_COLORS[getField(p, 'social_energy', 'socialEnergy')] || '#808080',
        getKey: (p) => getField(p, 'social_energy', 'socialEnergy'),
        colors: ENERGY_COLORS,
        formatLabel: (key) => key,
    },
    decision: {
        title: 'Decision Style',
        subtitle: 'Self-reported decision-making approach',
        getColor: (p) => DECISION_COLORS[getField(p, 'decision_style', 'decisionStyle')] || '#808080',
        getKey: (p) => getField(p, 'decision_style', 'decisionStyle'),
        colors: DECISION_COLORS,
        formatLabel: (key) => key,
    },
    region: {
        title: 'Geographic Region',
        subtitle: 'Self-reported location',
        getColor: (p) => REGION_COLORS[p.region] || '#808080',
        getKey: (p) => p.region,
        colors: REGION_COLORS,
        formatLabel: (key) => key,
    },
};

// =============================================================================
// STATE
// =============================================================================

let points = [];
let mode = 'cluster';
let hoveredPoint = null;
let nearestNeighbors = [];
let showNeighbors = false;
const NEIGHBOR_COUNT = 5;

// Animation state for smooth transitions
let pointColors = new Map(); // Current colors
let targetColors = new Map(); // Target colors for transition
let colorTransitionProgress = 1;
let animationFrame = null;

// Canvas
let canvas, ctx;
let dpr = 1;
let canvasWidth, canvasHeight;
const padding = 60;

// =============================================================================
// UTILITIES
// =============================================================================

function getField(obj, ...names) {
    for (const name of names) {
        if (obj[name] !== undefined) return obj[name];
    }
    return undefined;
}

function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 128, g: 128, b: 128 };
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = Math.round(x).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

function lerpColor(color1, color2, t) {
    const rgb1 = hexToRgb(color1);
    const rgb2 = hexToRgb(color2);
    return rgbToHex(
        rgb1.r + (rgb2.r - rgb1.r) * t,
        rgb1.g + (rgb2.g - rgb1.g) * t,
        rgb1.b + (rgb2.b - rgb1.b) * t
    );
}

function distance(p1, p2) {
    return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function truncate(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
}

// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    setupCanvas();
    await loadData();
    initializeColors();
    setupEventListeners();
    updateLegend();
    render();
}

function setupCanvas() {
    dpr = window.devicePixelRatio || 1;
    canvasWidth = window.innerWidth;
    canvasHeight = window.innerHeight;

    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;
    canvas.style.width = canvasWidth + 'px';
    canvas.style.height = canvasHeight + 'px';

    ctx.scale(dpr, dpr);
}

async function loadData() {
    try {
        const response = await fetch('points.json');
        points = await response.json();

        // Update stats
        document.getElementById('point-count').textContent = `${points.length} points`;
        const clusters = new Set(points.map(p => p.cluster));
        document.getElementById('cluster-count').textContent = `${clusters.size} clusters`;

        console.log(`Loaded ${points.length} points in ${clusters.size} clusters`);
    } catch (err) {
        console.error('Failed to load points.json:', err);
        document.getElementById('point-count').textContent = 'Error loading';
    }
}

function initializeColors() {
    const config = MODE_CONFIG[mode];
    for (const point of points) {
        const color = config.getColor(point);
        pointColors.set(point.id, color);
        targetColors.set(point.id, color);
    }
}

// =============================================================================
// EVENT HANDLERS
// =============================================================================

function setupEventListeners() {
    // Window resize
    window.addEventListener('resize', () => {
        setupCanvas();
        render();
    });

    // Mouse events
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseleave', handleMouseLeave);

    // Touch events
    canvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    canvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    canvas.addEventListener('touchend', handleTouchEnd);

    // Mode toggle buttons
    document.querySelectorAll('.mode-btn[data-group="view"]').forEach(btn => {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    });

    // Neighbor toggle
    document.getElementById('btn-neighbors').addEventListener('click', toggleNeighbors);

    // Info modal
    document.getElementById('btn-info').addEventListener('click', () => {
        document.getElementById('info-modal').classList.remove('hidden');
    });
    document.getElementById('modal-close').addEventListener('click', () => {
        document.getElementById('info-modal').classList.add('hidden');
    });
    document.getElementById('info-modal').addEventListener('click', (e) => {
        if (e.target.id === 'info-modal') {
            document.getElementById('info-modal').classList.add('hidden');
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

function handleKeyboard(e) {
    // Don't handle if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
        case '1': setMode('cluster'); break;
        case '2': setMode('energy'); break;
        case '3': setMode('decision'); break;
        case '4': setMode('region'); break;
        case 'n':
        case 'N':
            toggleNeighbors();
            break;
        case '?':
            document.getElementById('info-modal').classList.toggle('hidden');
            break;
        case 'Escape':
            document.getElementById('info-modal').classList.add('hidden');
            break;
    }
}

function setMode(newMode) {
    if (mode === newMode) return;
    mode = newMode;

    // Update button states
    document.querySelectorAll('.mode-btn[data-group="view"]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Start color transition
    startColorTransition();
    updateLegend();
}

function toggleNeighbors() {
    showNeighbors = !showNeighbors;
    document.getElementById('btn-neighbors').classList.toggle('active', showNeighbors);

    if (showNeighbors && hoveredPoint) {
        updateNeighbors();
    } else {
        nearestNeighbors = [];
        document.getElementById('neighbor-info').classList.add('hidden');
    }
    render();
}

function handleMouseMove(e) {
    const rect = canvas.getBoundingClientRect();
    handlePointerMove(e.clientX - rect.left, e.clientY - rect.top);
}

function handleMouseLeave() {
    handlePointerLeave();
}

function handleTouchStart(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        handlePointerMove(touch.clientX - rect.left, touch.clientY - rect.top);
    }
}

function handleTouchMove(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches[0];
        handlePointerMove(touch.clientX - rect.left, touch.clientY - rect.top);
    }
}

function handleTouchEnd() {
    // Keep panel visible on touch devices until next touch
}

function handlePointerMove(mouseX, mouseY) {
    const threshold = 25;
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

    if (nearest !== hoveredPoint) {
        hoveredPoint = nearest;

        if (showNeighbors && hoveredPoint) {
            updateNeighbors();
        } else if (!hoveredPoint) {
            nearestNeighbors = [];
            document.getElementById('neighbor-info').classList.add('hidden');
        }

        render();
        updatePanel();
    }
}

function handlePointerLeave() {
    if (hoveredPoint) {
        hoveredPoint = null;
        nearestNeighbors = [];
        document.getElementById('neighbor-info').classList.add('hidden');
        render();
        updatePanel();
    }
}

// =============================================================================
// NEIGHBOR HIGHLIGHTING
// =============================================================================

function updateNeighbors() {
    if (!hoveredPoint) {
        nearestNeighbors = [];
        return;
    }

    // Calculate distances to all other points
    const distances = points
        .filter(p => p.id !== hoveredPoint.id)
        .map(p => ({
            point: p,
            dist: distance(hoveredPoint, p)
        }))
        .sort((a, b) => a.dist - b.dist);

    nearestNeighbors = distances.slice(0, NEIGHBOR_COUNT).map(d => d.point);

    // Update UI
    document.getElementById('neighbor-count').textContent = nearestNeighbors.length;
    document.getElementById('neighbor-info').classList.remove('hidden');
}

// =============================================================================
// COLOR TRANSITIONS
// =============================================================================

function startColorTransition() {
    const config = MODE_CONFIG[mode];

    // Set target colors
    for (const point of points) {
        targetColors.set(point.id, config.getColor(point));
    }

    colorTransitionProgress = 0;

    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }

    animateColorTransition();
}

function animateColorTransition() {
    colorTransitionProgress += 0.08; // Transition speed

    if (colorTransitionProgress >= 1) {
        colorTransitionProgress = 1;
        // Finalize colors
        for (const point of points) {
            pointColors.set(point.id, targetColors.get(point.id));
        }
        render();
        return;
    }

    // Interpolate colors
    for (const point of points) {
        const currentColor = pointColors.get(point.id);
        const targetColor = targetColors.get(point.id);
        const interpolated = lerpColor(currentColor, targetColor, 0.15);
        pointColors.set(point.id, interpolated);
    }

    render();
    animationFrame = requestAnimationFrame(animateColorTransition);
}

// =============================================================================
// COORDINATE TRANSFORMATION
// =============================================================================

function dataToScreen(x, y) {
    const effectiveWidth = canvasWidth - padding * 2;
    const effectiveHeight = canvasHeight - padding * 2;

    const screenX = padding + ((x + 1) / 2) * effectiveWidth;
    const screenY = padding + ((1 - y) / 2) * effectiveHeight;

    return [screenX, screenY];
}

// =============================================================================
// RENDERING
// =============================================================================

function render() {
    // Clear with gradient background
    const gradient = ctx.createRadialGradient(
        canvasWidth / 2, canvasHeight / 2, 0,
        canvasWidth / 2, canvasHeight / 2, Math.max(canvasWidth, canvasHeight) * 0.6
    );
    gradient.addColorStop(0, '#0e0e14');
    gradient.addColorStop(1, '#0a0a0f');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Sort points: neighbors and hovered on top
    const sortedPoints = [...points].sort((a, b) => {
        const aIsNeighbor = nearestNeighbors.includes(a);
        const bIsNeighbor = nearestNeighbors.includes(b);
        const aIsHovered = a === hoveredPoint;
        const bIsHovered = b === hoveredPoint;

        if (aIsHovered) return 1;
        if (bIsHovered) return -1;
        if (aIsNeighbor && !bIsNeighbor) return 1;
        if (!aIsNeighbor && bIsNeighbor) return -1;
        return 0;
    });

    // Draw connection lines to neighbors
    if (showNeighbors && hoveredPoint && nearestNeighbors.length > 0) {
        const [hx, hy] = dataToScreen(hoveredPoint.x, hoveredPoint.y);

        for (let i = 0; i < nearestNeighbors.length; i++) {
            const neighbor = nearestNeighbors[i];
            const [nx, ny] = dataToScreen(neighbor.x, neighbor.y);

            // Gradient line
            const lineGradient = ctx.createLinearGradient(hx, hy, nx, ny);
            const opacity = 0.3 - (i * 0.05);
            lineGradient.addColorStop(0, `rgba(78, 205, 196, ${opacity})`);
            lineGradient.addColorStop(1, `rgba(78, 205, 196, ${opacity * 0.3})`);

            ctx.beginPath();
            ctx.moveTo(hx, hy);
            ctx.lineTo(nx, ny);
            ctx.strokeStyle = lineGradient;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    }

    // Draw points
    for (const point of sortedPoints) {
        const isHovered = point === hoveredPoint;
        const isNeighbor = nearestNeighbors.includes(point);
        drawPoint(point, isHovered, isNeighbor);
    }
}

function drawPoint(point, isHovered, isNeighbor) {
    const [x, y] = dataToScreen(point.x, point.y);
    const color = pointColors.get(point.id) || MODE_CONFIG[mode].getColor(point);

    // Determine size
    let baseRadius = 4;
    if (isHovered) baseRadius = 7;
    else if (isNeighbor) baseRadius = 5;

    // Determine opacity
    let opacity = 1;
    if (showNeighbors && hoveredPoint && !isHovered && !isNeighbor) {
        opacity = 0.25;
    }

    // Draw outer glow
    const glowRadius = baseRadius * (isHovered ? 4 : isNeighbor ? 3.5 : 3);
    const glowOpacity = isHovered ? 0.4 : isNeighbor ? 0.35 : 0.2;

    const gradient = ctx.createRadialGradient(x, y, 0, x, y, glowRadius);
    gradient.addColorStop(0, color + Math.round(glowOpacity * opacity * 255).toString(16).padStart(2, '0'));
    gradient.addColorStop(1, color + '00');

    ctx.beginPath();
    ctx.arc(x, y, glowRadius, 0, Math.PI * 2);
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw main point
    ctx.globalAlpha = opacity;
    ctx.beginPath();
    ctx.arc(x, y, baseRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Draw rings for hovered/neighbor
    if (isHovered) {
        // Inner ring
        ctx.beginPath();
        ctx.arc(x, y, baseRadius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Outer pulse ring
        ctx.beginPath();
        ctx.arc(x, y, baseRadius + 9, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
        ctx.lineWidth = 1;
        ctx.stroke();
    } else if (isNeighbor) {
        ctx.beginPath();
        ctx.arc(x, y, baseRadius + 3, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(78, 205, 196, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

// =============================================================================
// PANEL UPDATES
// =============================================================================

function updatePanel() {
    const panel = document.getElementById('hover-panel');

    if (!hoveredPoint) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');

    // Header
    document.getElementById('panel-nickname').textContent = hoveredPoint.nickname || 'anonymous';
    document.getElementById('panel-id').textContent = hoveredPoint.id;

    // Questions
    const contentEl = document.getElementById('panel-content');
    const questions = parseQuestions(hoveredPoint.text);

    contentEl.innerHTML = questions.map(q => `
        <div class="question-block">
            <div class="question-label">${q.label}</div>
            <div class="question-text">${truncate(q.text, 200)}</div>
        </div>
    `).join('');

    // Footer metadata
    const footerEl = document.getElementById('panel-footer');
    const clusterColor = CLUSTER_COLORS[hoveredPoint.cluster % CLUSTER_COLORS.length];
    const decisionStyle = getField(hoveredPoint, 'decision_style', 'decisionStyle') || 'N/A';
    const socialEnergy = getField(hoveredPoint, 'social_energy', 'socialEnergy') || 'N/A';
    const region = hoveredPoint.region || 'N/A';

    footerEl.innerHTML = `
        <div class="meta-row">
            <span class="meta-label">Model cluster</span>
            <span class="meta-value">
                <span class="cluster-badge" style="background: ${clusterColor}; color: ${clusterColor}"></span>
                ${hoveredPoint.cluster}
            </span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Decision style</span>
            <span class="meta-value">${decisionStyle}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Social energy</span>
            <span class="meta-value">${socialEnergy}</span>
        </div>
        <div class="meta-row">
            <span class="meta-label">Region</span>
            <span class="meta-value">${region}</span>
        </div>
    `;
}

function updateLegend() {
    const config = MODE_CONFIG[mode];
    const titleEl = document.getElementById('legend-title');
    const subtitleEl = document.getElementById('legend-subtitle');
    const itemsEl = document.getElementById('legend-items');

    titleEl.textContent = config.title;
    subtitleEl.textContent = config.subtitle;

    // Collect unique values and counts
    const counts = new Map();
    for (const point of points) {
        const key = config.getKey(point);
        counts.set(key, (counts.get(key) || 0) + 1);
    }

    // Sort by count (descending) or by key for clusters
    let sortedKeys = [...counts.keys()];
    if (mode === 'cluster') {
        sortedKeys.sort((a, b) => a - b);
    } else {
        sortedKeys.sort((a, b) => counts.get(b) - counts.get(a));
    }

    itemsEl.innerHTML = sortedKeys.map(key => {
        const color = mode === 'cluster'
            ? CLUSTER_COLORS[key % CLUSTER_COLORS.length]
            : config.colors[key] || '#808080';
        const count = counts.get(key);
        const label = config.formatLabel(key);

        return `
            <div class="legend-item">
                <span class="legend-dot" style="background: ${color}; color: ${color}"></span>
                <span>${label}</span>
                <span class="legend-count">${count}</span>
            </div>
        `;
    }).join('');
}

// =============================================================================
// QUESTION PARSING
// =============================================================================

function parseQuestions(text) {
    const questions = [];
    const labelPatterns = [
        { pattern: /Q1 \(([^)]+)\):/, fallback: 'Safe Place' },
        { pattern: /Q2 \(([^)]+)\):/, fallback: 'Handling Stress' },
        { pattern: /Q3 \(([^)]+)\):/, fallback: 'Feeling Understood' },
        { pattern: /Q4 \(([^)]+)\):/, fallback: 'Free Day' },
        { pattern: /Q5 \(([^)]+)\):/, fallback: 'One Word' },
    ];

    const lines = text.split('\n');

    for (const line of lines) {
        for (const { pattern, fallback } of labelPatterns) {
            const match = line.match(pattern);
            if (match) {
                const label = match[1] || fallback;
                const textContent = line.replace(pattern, '').trim();
                questions.push({ label, text: textContent });
                break;
            }
        }
    }

    // Fallback if no patterns matched
    if (questions.length === 0) {
        const fallbackLabels = ['Response 1', 'Response 2', 'Response 3', 'Response 4', 'Response 5'];
        text.split('\n').filter(l => l.trim()).forEach((part, i) => {
            questions.push({
                label: fallbackLabels[i] || `Response ${i + 1}`,
                text: part.trim()
            });
        });
    }

    return questions;
}

// =============================================================================
// SELF-SUBMISSION FEATURE
// =============================================================================

let selfPoint = null;
let embeddingPipeline = null;
let isLoadingModel = false;

function setupSelfSubmission() {
    const btnAddSelf = document.getElementById('btn-add-self');
    const submitModal = document.getElementById('submit-modal');
    const submitModalClose = document.getElementById('submit-modal-close');
    const cancelSubmit = document.getElementById('cancel-submit');
    const selfForm = document.getElementById('self-form');

    btnAddSelf.addEventListener('click', () => {
        submitModal.classList.remove('hidden');
    });

    submitModalClose.addEventListener('click', () => {
        submitModal.classList.add('hidden');
    });

    cancelSubmit.addEventListener('click', () => {
        submitModal.classList.add('hidden');
    });

    submitModal.addEventListener('click', (e) => {
        if (e.target === submitModal) {
            submitModal.classList.add('hidden');
        }
    });

    selfForm.addEventListener('submit', handleSelfSubmit);

    // Space key to open submission
    document.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && !e.target.matches('input, textarea, select')) {
            e.preventDefault();
            submitModal.classList.remove('hidden');
        }
    });
}

async function handleSelfSubmit(e) {
    e.preventDefault();

    const form = e.target;
    const formData = new FormData(form);

    // Show processing state
    const processingStatus = document.getElementById('processing-status');
    const findPlaceBtn = document.getElementById('find-place-btn');
    const btnText = findPlaceBtn.querySelector('.btn-text');
    const btnLoading = findPlaceBtn.querySelector('.btn-loading');

    btnText.classList.add('hidden');
    btnLoading.classList.remove('hidden');
    findPlaceBtn.disabled = true;
    processingStatus.classList.remove('hidden');

    try {
        // Build user text from responses
        const userText = buildUserText(formData);
        const metadata = extractMetadata(formData);

        updateProcessingStep('Loading embedding model...');
        await loadEmbeddingModel();

        updateProcessingStep('Generating your embedding...');
        const userEmbedding = await generateEmbedding(userText);

        updateProcessingStep('Finding your position...');
        const position = await findPosition(userEmbedding, userText);

        updateProcessingStep('Determining your cluster...');
        const cluster = findNearestCluster(position);

        // Create the self point
        selfPoint = {
            id: 'self',
            text: userText,
            x: position.x,
            y: position.y,
            cluster: cluster,
            decision_style: metadata.decision_style,
            social_energy: metadata.social_energy,
            region: metadata.region,
            nickname: metadata.nickname || 'You',
            isSelf: true
        };

        // Add to points array
        points.push(selfPoint);

        // Initialize colors for the new point
        const config = MODE_CONFIG[mode];
        const color = config.getColor(selfPoint);
        pointColors.set(selfPoint.id, color);
        targetColors.set(selfPoint.id, color);

        // Update stats
        document.getElementById('point-count').textContent = `${points.length} points`;

        // Close modal and render
        document.getElementById('submit-modal').classList.add('hidden');

        // Show position indicator
        showYourPosition(cluster);

        // Re-render
        render();

        // Update legend
        updateLegend();

        // Highlight the self point briefly
        setTimeout(() => {
            hoveredPoint = selfPoint;
            if (showNeighbors) updateNeighbors();
            render();
            updatePanel();
        }, 300);

    } catch (err) {
        console.error('Self-submission error:', err);
        updateProcessingStep(`Error: ${err.message}`);
    } finally {
        btnText.classList.remove('hidden');
        btnLoading.classList.add('hidden');
        findPlaceBtn.disabled = false;
        setTimeout(() => processingStatus.classList.add('hidden'), 2000);
    }
}

function buildUserText(formData) {
    const parts = [
        `Q1 (safe place): ${formData.get('q1_safe_place')}`,
        `Q2 (stress): ${formData.get('q2_stress')}`,
        `Q3 (understood): ${formData.get('q3_understood')}`,
        `Q4 (free day): ${formData.get('q4_free_day')}`,
        `Q5 (one word): ${formData.get('q5_one_word')}`,
    ];
    return parts.join('\n');
}

function extractMetadata(formData) {
    return {
        decision_style: formData.get('q6_decision_style'),
        social_energy: formData.get('q7_social_energy'),
        region: formData.get('q8_region'),
        nickname: formData.get('nickname')
    };
}

function updateProcessingStep(text) {
    document.getElementById('processing-step').textContent = text;
}

async function loadEmbeddingModel() {
    if (embeddingPipeline) return;
    if (isLoadingModel) {
        // Wait for existing load
        while (isLoadingModel) {
            await new Promise(r => setTimeout(r, 100));
        }
        return;
    }

    isLoadingModel = true;
    try {
        // Import Transformers.js
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');

        // Use a small, fast model for browser
        embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            quantized: true,
            progress_callback: (progress) => {
                if (progress.status === 'downloading') {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    updateProcessingStep(`Downloading model... ${pct}%`);
                }
            }
        });
    } finally {
        isLoadingModel = false;
    }
}

async function generateEmbedding(text) {
    const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
}

async function findPosition(userEmbedding, userText) {
    // Strategy: Use text similarity to find position
    // Since we don't have original embeddings stored, we'll use a simple approach:
    // Generate embeddings for a sample of existing points and interpolate

    // For performance, sample points if there are many
    const sampleSize = Math.min(points.length, 30);
    const sampledPoints = points.length <= sampleSize
        ? points
        : sampleRandomPoints(points, sampleSize);

    // Generate embeddings for sampled points
    const pointEmbeddings = [];
    for (let i = 0; i < sampledPoints.length; i++) {
        const point = sampledPoints[i];
        updateProcessingStep(`Comparing with point ${i + 1}/${sampledPoints.length}...`);
        const embedding = await generateEmbedding(point.text);
        pointEmbeddings.push({ point, embedding });
    }

    // Calculate cosine similarities
    const similarities = pointEmbeddings.map(({ point, embedding }) => ({
        point,
        similarity: cosineSimilarity(userEmbedding, embedding)
    }));

    // Sort by similarity
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Use weighted average of top k neighbors
    const k = Math.min(5, similarities.length);
    let totalWeight = 0;
    let x = 0;
    let y = 0;

    for (let i = 0; i < k; i++) {
        const { point, similarity } = similarities[i];
        // Convert similarity to weight (higher similarity = more weight)
        const weight = Math.pow(Math.max(0, similarity), 2);
        x += point.x * weight;
        y += point.y * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0) {
        x /= totalWeight;
        y /= totalWeight;
    } else {
        // Fallback: center position with small random offset
        x = (Math.random() - 0.5) * 0.2;
        y = (Math.random() - 0.5) * 0.2;
    }

    // Add small jitter to avoid exact overlap
    x += (Math.random() - 0.5) * 0.02;
    y += (Math.random() - 0.5) * 0.02;

    return { x, y };
}

function sampleRandomPoints(arr, n) {
    const shuffled = [...arr].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, n);
}

function cosineSimilarity(a, b) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function findNearestCluster(position) {
    // Find the cluster of the nearest existing point
    let nearestDist = Infinity;
    let nearestCluster = 0;

    for (const point of points) {
        if (point.isSelf) continue;
        const dist = Math.hypot(position.x - point.x, position.y - point.y);
        if (dist < nearestDist) {
            nearestDist = dist;
            nearestCluster = point.cluster;
        }
    }

    return nearestCluster;
}

function showYourPosition(cluster) {
    const positionEl = document.getElementById('your-position');
    const clusterEl = document.getElementById('your-cluster');

    clusterEl.textContent = cluster;
    positionEl.classList.remove('hidden');

    // Position the indicator near the self point
    updateYourPositionIndicator();
}

function updateYourPositionIndicator() {
    if (!selfPoint) return;

    const positionEl = document.getElementById('your-position');
    const [screenX, screenY] = dataToScreen(selfPoint.x, selfPoint.y);

    // Position above the point
    positionEl.style.left = `${screenX}px`;
    positionEl.style.top = `${screenY - 50}px`;
}

// Update position indicator on resize
window.addEventListener('resize', () => {
    if (selfPoint) {
        updateYourPositionIndicator();
    }
});

// Override drawPoint to show special styling for self point
const originalDrawPoint = drawPoint;
function drawPointWithSelf(point, isHovered, isNeighbor) {
    if (point.isSelf) {
        const [x, y] = dataToScreen(point.x, point.y);
        const color = pointColors.get(point.id) || MODE_CONFIG[mode].getColor(point);

        // Draw pulsing ring animation
        const time = Date.now() / 1000;
        const pulseScale = 1 + Math.sin(time * 2) * 0.2;

        // Outer glow
        const gradient = ctx.createRadialGradient(x, y, 0, x, y, 25 * pulseScale);
        gradient.addColorStop(0, color + '60');
        gradient.addColorStop(0.5, color + '30');
        gradient.addColorStop(1, color + '00');
        ctx.beginPath();
        ctx.arc(x, y, 25 * pulseScale, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Main point (larger)
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // White ring
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.stroke();

        // Outer ring
        ctx.beginPath();
        ctx.arc(x, y, 18 * pulseScale, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();

    } else {
        originalDrawPoint(point, isHovered, isNeighbor);
    }
}

// Replace drawPoint
drawPoint = drawPointWithSelf;

// Animate self point
function animateSelfPoint() {
    if (selfPoint) {
        render();
    }
    requestAnimationFrame(animateSelfPoint);
}

// =============================================================================
// START
// =============================================================================

init();
setupSelfSubmission();
animateSelfPoint();
