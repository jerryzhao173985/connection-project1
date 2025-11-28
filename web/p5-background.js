/**
 * Hidden Connections - p5.js Background Animation
 * Flowing particle field creating ambient atmosphere
 */

// Skip animation on mobile/touch devices for performance
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Particles for ambient animation
let particles = [];
let flowField;
let cols, rows;
const scl = 30;
const particleCount = isTouchDevice ? 0 : 150; // No particles on mobile
let zoff = 0;

// Colors matching the theme
const bgColor = [10, 10, 15];
const particleColors = [
    [78, 205, 196, 40],   // teal
    [255, 107, 107, 35],  // coral
    [201, 177, 255, 35],  // lavender
    [255, 230, 109, 30],  // yellow
    [149, 225, 211, 35],  // mint
];

function setup() {
    // Skip setup entirely on mobile/touch or reduced motion
    if (isTouchDevice || prefersReducedMotion) {
        noCanvas();
        return;
    }

    // Create canvas in the container
    const container = document.getElementById('p5-container');
    const canvas = createCanvas(windowWidth, windowHeight);
    canvas.parent(container);

    // Setup flow field
    cols = floor(width / scl);
    rows = floor(height / scl);
    flowField = new Array(cols * rows);

    // Create particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    // Slow down the animation
    frameRate(30);
}

function draw() {
    // Skip drawing on mobile/touch
    if (isTouchDevice || prefersReducedMotion) {
        return;
    }

    // Semi-transparent background for trails
    background(bgColor[0], bgColor[1], bgColor[2], 25);

    // Update flow field with Perlin noise
    let yoff = 0;
    for (let y = 0; y < rows; y++) {
        let xoff = 0;
        for (let x = 0; x < cols; x++) {
            const index = x + y * cols;
            const angle = noise(xoff, yoff, zoff) * TWO_PI * 2;
            const v = p5.Vector.fromAngle(angle);
            v.setMag(0.5);
            flowField[index] = v;
            xoff += 0.1;
        }
        yoff += 0.1;
    }
    zoff += 0.002;

    // Update and show particles
    for (let particle of particles) {
        particle.follow(flowField);
        particle.update();
        particle.edges();
        particle.show();
    }

    // Draw subtle connection lines between close particles
    stroke(255, 255, 255, 8);
    strokeWeight(0.5);
    for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
            const d = dist(particles[i].pos.x, particles[i].pos.y,
                          particles[j].pos.x, particles[j].pos.y);
            if (d < 80) {
                const alpha = map(d, 0, 80, 15, 0);
                stroke(255, 255, 255, alpha);
                line(particles[i].pos.x, particles[i].pos.y,
                     particles[j].pos.x, particles[j].pos.y);
            }
        }
    }
}

function windowResized() {
    resizeCanvas(windowWidth, windowHeight);
    cols = floor(width / scl);
    rows = floor(height / scl);
    flowField = new Array(cols * rows);
}

// Particle class
class Particle {
    constructor() {
        this.pos = createVector(random(width), random(height));
        this.vel = createVector(0, 0);
        this.acc = createVector(0, 0);
        this.maxSpeed = 1.5;
        this.prevPos = this.pos.copy();
        this.color = random(particleColors);
        this.size = random(1.5, 3);
    }

    follow(vectors) {
        const x = floor(this.pos.x / scl);
        const y = floor(this.pos.y / scl);
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
        if (this.pos.x > width) {
            this.pos.x = 0;
            this.prevPos.x = 0;
        }
        if (this.pos.x < 0) {
            this.pos.x = width;
            this.prevPos.x = width;
        }
        if (this.pos.y > height) {
            this.pos.y = 0;
            this.prevPos.y = 0;
        }
        if (this.pos.y < 0) {
            this.pos.y = height;
            this.prevPos.y = height;
        }
    }

    show() {
        // Draw glow
        noStroke();
        fill(this.color[0], this.color[1], this.color[2], this.color[3] * 0.5);
        ellipse(this.pos.x, this.pos.y, this.size * 4, this.size * 4);

        // Draw core
        fill(this.color[0], this.color[1], this.color[2], this.color[3] * 1.5);
        ellipse(this.pos.x, this.pos.y, this.size, this.size);

        this.prevPos.x = this.pos.x;
        this.prevPos.y = this.pos.y;
    }
}
