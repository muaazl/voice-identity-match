/**
 * VoiceMimic Audio Recorder & Waveform Visualizer.
 * Captures browser microphone, resamples to 16kHz mono, and encodes to standard 16-bit PCM WAV.
 * Features a high-DPI responsive organic audio waveform visualizer.
 */

class AudioRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.source = null;
        this.recordedSamples = [];
        this.recording = false;
        this.targetSampleRate = 16000;
        this.visualizerCanvas = null;
        this.animationId = null;
        this.currentRms = 0;
        this.targetRms = 0;
    }

    /**
     * Start recording mono audio at 16kHz.
     * @param {HTMLCanvasElement|null} canvas - Optional canvas for live waveform.
     */
    async start(canvas = null) {
        if (this.recording) return;

        this.recordedSamples = [];
        this.visualizerCanvas = canvas;
        this.currentRms = 0;
        this.targetRms = 0;

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: false,
            },
        });
        this.mediaStream = stream;

        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx();
        this.nativeSampleRate = this.audioContext.sampleRate;

        this.source = this.audioContext.createMediaStreamSource(stream);
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

        this.processor.onaudioprocess = (e) => {
            if (!this.recording) return;

            const inputData = e.inputBuffer.getChannelData(0);
            // Collect pristine samples at native rate without per-chunk truncation
            this.recordedSamples.push(new Float32Array(inputData));

            // Instantaneous RMS computation
            let sumSquare = 0;
            for (let i = 0; i < inputData.length; i++) {
                sumSquare += inputData[i] * inputData[i];
            }
            this.targetRms = Math.sqrt(sumSquare / inputData.length);
        };

        this.source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        this.recording = true;

        if (this.visualizerCanvas) {
            this._setupCanvasResolution();
            this._startWaveformAnimation();
        }
    }

    /**
     * Stop recording and return 16-bit 16kHz mono WAV Blob.
     * @returns {Promise<Blob>}
     */
    async stop() {
        if (!this.recording) return null;
        this.recording = false;

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }

        if (this.visualizerCanvas) {
            this._clearWaveformCanvas();
        }

        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        if (this.source) {
            this.source.disconnect();
            this.source = null;
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
            this.audioContext = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach((track) => track.stop());
            this.mediaStream = null;
        }

        let totalSamples = 0;
        for (const chunk of this.recordedSamples) {
            totalSamples += chunk.length;
        }

        const mergedNativeSamples = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of this.recordedSamples) {
            mergedNativeSamples.set(chunk, offset);
            offset += chunk.length;
        }

        // Perform hardware-accelerated bandlimited anti-aliased sinc resampling to 16kHz
        const resampledSamples = await this._resampleTo16k(mergedNativeSamples, this.nativeSampleRate || 48000);

        return this._encodeWav(resampledSamples, this.targetSampleRate);
    }

    isRecording() {
        return this.recording;
    }

    /** Resample float array to 16000Hz using browser OfflineAudioContext or fallback */
    async _resampleTo16k(samples, inputRate) {
        if (inputRate === this.targetSampleRate || samples.length === 0) return samples;

        const targetLength = Math.max(1, Math.round((samples.length / inputRate) * this.targetSampleRate));
        const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;

        if (OfflineCtx) {
            try {
                const offlineCtx = new OfflineCtx(1, targetLength, this.targetSampleRate);
                const audioBuffer = offlineCtx.createBuffer(1, samples.length, inputRate);
                audioBuffer.getChannelData(0).set(samples);

                const source = offlineCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(offlineCtx.destination);
                source.start(0);

                const rendered = await offlineCtx.startRendering();
                return rendered.getChannelData(0);
            } catch (err) {
                console.warn('OfflineAudioContext resample failed, using fallback:', err);
            }
        }

        return this._resampleAudio(samples, inputRate, this.targetSampleRate);
    }

    /** Linear interpolation fallback with boundary preservation */
    _resampleAudio(inputData, inputRate, outputRate) {
        if (inputRate === outputRate) return inputData;

        const ratio = inputRate / outputRate;
        const newLength = Math.round(inputData.length / ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const originIndex = i * ratio;
            const indexBefore = Math.floor(originIndex);
            const indexAfter = Math.min(indexBefore + 1, inputData.length - 1);
            const weight = originIndex - indexBefore;
            result[i] = inputData[indexBefore] * (1 - weight) + inputData[indexAfter] * weight;
        }
        return result;
    }

    /** Encode Float32Array into standard 44-byte RIFF/WAVE 16-bit PCM format */
    _encodeWav(samples, sampleRate) {
        const numChannels = 1;
        const bitsPerSample = 16;
        const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
        const blockAlign = (numChannels * bitsPerSample) / 8;
        const dataLength = samples.length * (bitsPerSample / 8);
        const buffer = new ArrayBuffer(44 + dataLength);
        const view = new DataView(buffer);

        // RIFF chunk descriptor
        this._writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + dataLength, true);
        this._writeString(view, 8, 'WAVE');

        // fmt subchunk
        this._writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);

        // data subchunk
        this._writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        // 16-bit PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    _writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /** Ensure high-DPI crisp rendering on Retina / 4K displays */
    _setupCanvasResolution() {
        if (!this.visualizerCanvas) return;
        const canvas = this.visualizerCanvas;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const width = rect.width || 320;
        const height = rect.height || 48;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        const ctx = canvas.getContext('2d');
        ctx.resetTransform();
        ctx.scale(dpr, dpr);
    }

    /** Clean, fluid mirrored energy visualizer */
    _startWaveformAnimation() {
        const canvas = this.visualizerCanvas;
        const ctx = canvas.getContext('2d');
        let phase = 0;

        const render = () => {
            if (!this.recording) return;

            const rect = canvas.getBoundingClientRect();
            const width = rect.width || 320;
            const height = rect.height || 48;

            ctx.clearRect(0, 0, width, height);

            // Smooth RMS interpolation
            this.currentRms += (this.targetRms - this.currentRms) * 0.25;

            const midY = height / 2;
            const numBars = 32;
            const barSpacing = width / numBars;
            const barWidth = Math.max(2.5, barSpacing * 0.55);

            for (let i = 0; i < numBars; i++) {
                const norm = i / (numBars - 1);
                // Bell curve envelope so edges taper gracefully
                const envelope = Math.sin(norm * Math.PI);
                const wave = Math.sin(i * 0.35 + phase) * 0.35 + 0.65;
                const dynamicHeight = Math.max(4, this.currentRms * height * 2.6 * envelope * wave);

                const x = i * barSpacing + (barSpacing - barWidth) / 2;
                const topY = midY - dynamicHeight / 2;

                // Clean indigo gradient
                const gradient = ctx.createLinearGradient(0, topY, 0, topY + dynamicHeight);
                gradient.addColorStop(0, '#4F46E5'); // Indigo 600
                gradient.addColorStop(1, '#818CF8'); // Indigo 400

                ctx.fillStyle = gradient;
                this._drawRoundedRect(ctx, x, topY, barWidth, dynamicHeight, barWidth / 2);
            }

            phase += 0.12;
            this.animationId = requestAnimationFrame(render);
        };

        render();
    }

    _clearWaveformCanvas() {
        if (!this.visualizerCanvas) return;
        const canvas = this.visualizerCanvas;
        const ctx = canvas.getContext('2d');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width || 320;
        const height = rect.height || 48;

        ctx.clearRect(0, 0, width, height);

        // Calm, resting flat line
        ctx.beginPath();
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 1.5;
        ctx.moveTo(8, height / 2);
        ctx.lineTo(width - 8, height / 2);
        ctx.stroke();
    }

    _drawRoundedRect(ctx, x, y, width, height, radius) {
        const r = Math.min(radius, width / 2, height / 2);
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + width, y, x + width, y + height, r);
        ctx.arcTo(x + width, y + height, x, y + height, r);
        ctx.arcTo(x, y + height, x, y, r);
        ctx.arcTo(x, y, x + width, y, r);
        ctx.closePath();
        ctx.fill();
    }
}

window.AudioRecorder = AudioRecorder;
