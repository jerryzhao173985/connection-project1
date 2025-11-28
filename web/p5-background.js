/**
 * Hidden Connections - p5.js Background Animation
 * Adaptive flowing particle field that works on all devices
 */

// Device detection
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const isLowPowerMode = navigator.connection?.saveData || false;

// Adaptive settings based on device
function getSettings() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    const pixelRatio = window.devicePixelRatio || 1;
    const area = width * height;

    // Determine device tier
    let tier = 'high'; // Desktop
    if (prefersReducedMotion || isLowPowerMode) {
        tier = 'minimal';
    } else if (isMobile && width < 768) {
        tier = 'low';
    } else if (isMobile || width < 1024) {
        tier = 'medium';
    }

    const settings = {
        minimal: {
            particleCount: 0,
            frameRate: 15,
            connectionDistance: 0,
            scale: 40,
            opacity: 0.3
        },
        low: {
            particleCount: Math.floor(area / 25000), // ~30 on phone
            frameRate: 20,
            connectionDistance: 60,
            scale: 40,
            opacity: 0.5
        },
        medium: {
            particleCount: Math.floor(area / 15000), // ~60 on tablet
            frameRate: 24,
            connectionDistance: 70,
            scale: 35,
            opacity: 0.6
        },
        high: {
            particleCount: Math.floor(area / 10000), // ~150 on desktop
            frameRate: 30,
            connectionDistance: 80,
            scale: 30,
            opacity: 0.7
        }
    };

    // Clamp particle count
    const s = settings[tier];
    s.particleCount = Math.max(0, Math.min(200, s.particleCount));

    return s;
}

// State
let particles = [];
let flowField;
let cols, rows;
let zoff = 0;
let settings;
let isActive = true;

// Colors matching the theme
const particleColors = [
    [78, 205, 196],   // teal
    [255, 107, 107],  // coral
    [201, 177, 255],  // lavender
    [255, 230, 109],  // yellow
    [149, 225, 211],  // mint
    [116, 185, 255],  // sky blue
];

function setup() {
    settings = getSettings();

    // Skip entirely if no particles
    if (settings.particleCount === 0) {
        noCanvas();
        return;
    }

    // Create canvas
    const container = document.getElementById('p5-container');
    if (!container) {
        noCanvas();
        return;
    }

    const canvas = createCanvas(windowWidth, windowHeight);
    canvas.parent(container);

    // Use lower pixel density on mobile for performance
    if (isMobile) {
        pixelDensity(1);
    }

    // Setup flow field grid
    cols = floor(width / settings.scale);
    rows = floor(height / settings.scale);
    flowField = new Array(cols * rows);

    // Create particles
    createParticles();

    // Set frame rate
    frameRate(settings.frameRate);

    // Handle visibility changes to pause when tab is hidden
    document.addEventListener('visibilitychange', () => {
        isActive = document.visibilityState === 'visible';
        if (isActive) {
            loop();
        } else {
            noLoop();
        }
    });
}

function createParticles() {
    particles = [];
    for (let i = 0; i < settings.particleCount; i++) {
        particles.push(new Particle());
    }
}

function draw() {
    if (!isActive || settings.particleCount === 0) return;

    // Semi-transparent background for trails
    background(10, 10, 15, 20);

    // Update flow field with Perlin noise
    let yoff = 0;
    for (let y = 0; y < rows; y++) {
        let xoff = 0;
        for (let x = 0; x < cols; x++) {
            const index = x + y * cols;
            const angle = noise(xoff, yoff, zoff) * TWO_PI * 2;
            const v = p5.Vector.fromAngle(angle);
            v.setMag(0.4);
            flowField[index] = v;
            xoff += 0.08;
        }
        yoff += 0.08;
    }
    zoff += 0.0015;

    // Update and show particles
    for (let particle of particles) {
        particle.follow(flowField);
        particle.update();
        particle.edges();
        particle.show();
    }

    // Draw connection lines (only if enabled and not too many particles)
    if (settings.connectionDistance > 0 && particles.length < 80) {
        drawConnections();
    }
}

function drawConnections() {
    const maxDist = settings.connectionDistance;

    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const d = dist(
                particles[i].pos.x, particles[i].pos.y,
                particles[j].pos.x, particles[j].pos.y
            );
            if (d < maxDist) {
                const alpha = map(d, 0, maxDist, 12, 0);
                stroke(255, 255, 255, alpha);
                strokeWeight(0.5);
                line(
                    particles[i].pos.x, particles[i].pos.y,
                    particles[j].pos.x, particles[j].pos.y
                );
            }
        }
    }
}

function windowResized() {
    if (settings.particleCount === 0) return;

    resizeCanvas(windowWidth, windowHeight);

    // Recalculate settings on resize
    const newSettings = getSettings();

    // Only recreate if particle count changed significantly
    if (Math.abs(newSettings.particleCount - settings.particleCount) > 20) {
        settings = newSettings;
        createParticles();
    }

    // Always update grid
    cols = floor(width / settings.scale);
    rows = floor(height / settings.scale);
    flowField = new Array(cols * rows);
}

// Particle class
class Particle {
    constructor() {
        this.pos = createVector(random(width), random(height));
        this.vel = createVector(random(-0.5, 0.5), random(-0.5, 0.5));
        this.acc = createVector(0, 0);
        this.maxSpeed = 1.2;
        this.color = random(particleColors);
        this.size = random(1.5, 3);
        this.alpha = random(25, 45);
    }

    follow(vectors) {
        if (!vectors || vectors.length === 0) return;

        const x = floor(this.pos.x / settings.scale);
        const y = floor(this.pos.y / settings.scale);
        const index = constrain(x + y * cols, 0, vectors.length - 1);
        const force = vectors[index];

        if (force) {
            this.applyForce(force);
        }
    }

    applyForce(force) {
        this.acc.add(force);
    }

    update() {
        this.vel.add(this.acc);
        this.vel.limit(this.maxSpeed);
        this.pos.add(this.vel);
        this.acc.mult(0);
    }

    edges() {
        if (this.pos.x > width) this.pos.x = 0;
        if (this.pos.x < 0) this.pos.x = width;
        if (this.pos.y > height) this.pos.y = 0;
        if (this.pos.y < 0) this.pos.y = height;
    }

    show() {
        noStroke();

        // Glow effect
        const glowAlpha = this.alpha * 0.4;
        fill(this.color[0], this.color[1], this.color[2], glowAlpha);
        ellipse(this.pos.x, this.pos.y, this.size * 4, this.size * 4);

        // Core
        fill(this.color[0], this.color[1], this.color[2], this.alpha);
        ellipse(this.pos.x, this.pos.y, this.size, this.size);
    }
}
