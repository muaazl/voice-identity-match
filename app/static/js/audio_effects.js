/**
 * VoiceMimic Audio Effects Synthesizer.
 * Generates subtle, pleasant UI sound effects via Web Audio API (no external asset files).
 */

class SoundEffects {
    constructor() {
        this.ctx = null;
        this.enabled = localStorage.getItem('vm_sound_enabled') !== 'false';
    }

    _getCtx() {
        if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {
                this.ctx = new AudioCtx();
            }
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
        return this.ctx;
    }

    toggle() {
        this.enabled = !this.enabled;
        localStorage.setItem('vm_sound_enabled', this.enabled.toString());
        if (this.enabled) {
            this.click();
        }
        return this.enabled;
    }

    isEnabled() {
        return this.enabled;
    }

    click() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(400, ctx.currentTime + 0.04);

        gain.gain.setValueAtTime(0.08, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.04);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.04);
    }

    recordStart() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(440, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.12);

        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.12);
    }

    recordStop() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(660, ctx.currentTime);
        osc.frequency.exponentialRampToValueAtTime(330, ctx.currentTime + 0.1);

        gain.gain.setValueAtTime(0.1, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.1);
    }

    tick() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(950, ctx.currentTime);

        gain.gain.setValueAtTime(0.04, ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.03);

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.start();
        osc.stop(ctx.currentTime + 0.03);
    }

    successChime() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        [523.25, 659.25, 783.99, 1046.5].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, now + i * 0.07);

            gain.gain.setValueAtTime(0.09, now + i * 0.07);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.07 + 0.28);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(now + i * 0.07);
            osc.stop(now + i * 0.07 + 0.28);
        });
    }

    failureTone() {
        if (!this.enabled) return;
        const ctx = this._getCtx();
        if (!ctx) return;

        const now = ctx.currentTime;
        [280, 220].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, now + i * 0.12);

            gain.gain.setValueAtTime(0.08, now + i * 0.12);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + i * 0.12 + 0.2);

            osc.connect(gain);
            gain.connect(ctx.destination);

            osc.start(now + i * 0.12);
            osc.stop(now + i * 0.12 + 0.2);
        });
    }
}

window.SoundEffects = SoundEffects;
