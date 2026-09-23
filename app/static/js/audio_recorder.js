/**
 * VoiceMimic Audio Recorder & Single-Line Waveform Visualizer.
 * Captures browser microphone, resamples to 16kHz mono, and encodes to 16-bit PCM WAV.
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
    }

    /**
     * Start recording mono audio at 16kHz.
     * @param {HTMLCanvasElement|null} canvas - Optional canvas for minimalist single-line wave.
     */
    async start(canvas = null) {
        if (this.recording) return;

        this.recordedSamples = [];
        this.visualizerCanvas = canvas;

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: false,
                autoGainControl: true,
            },
        });
        this.mediaStream = stream;

        // Create AudioContext (match browser native sample rate first, resample in processor)
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioCtx();
        const nativeSampleRate = this.audioContext.sampleRate;

        this.source = this.audioContext.createMediaStreamSource(stream);

        // Buffer size 4096 gives smooth processing without audio dropouts
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

        this.processor.onaudioprocess = (e) => {
            if (!this.recording) return;

            const inputData = e.inputBuffer.getChannelData(0);

            // Resample down to 16000 Hz if native rate differs
            const resampled = this._resampleAudio(inputData, nativeSampleRate, this.targetSampleRate);
            this.recordedSamples.push(new Float32Array(resampled));

            // Calculate instantaneous RMS for minimal visual feedback
            let sumSquare = 0;
            for (let i = 0; i < inputData.length; i++) {
                sumSquare += inputData[i] * inputData[i];
            }
            const rms = Math.sqrt(sumSquare / inputData.length);
            // Smooth RMS
            this.currentRms = this.currentRms * 0.7 + rms * 0.3;
        };

        this.source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        this.recording = true;

        if (this.visualizerCanvas) {
            this._startWaveformAnimation();
        }
    }

    /**
     * Stop recording and return a 16-bit 16kHz mono WAV Blob.
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

        // Concatenate all recorded Float32 chunks
        let totalSamples = 0;
        for (const chunk of this.recordedSamples) {
            totalSamples += chunk.length;
        }

        const mergedSamples = new Float32Array(totalSamples);
        let offset = 0;
        for (const chunk of this.recordedSamples) {
            mergedSamples.set(chunk, offset);
            offset += chunk.length;
        }

        // Encode to 16-bit PCM WAV Blob
        return this._encodeWav(mergedSamples, this.targetSampleRate);
    }

    isRecording() {
        return this.recording;
    }

    /** Resample float audio array from native rate to target rate (16000Hz) */
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
            // Linear interpolation
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
        view.setUint32(16, 16, true); // Subchunk1Size for PCM
        view.setUint16(20, 1, true); // AudioFormat: 1 = PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);

        // data subchunk
        this._writeString(view, 36, 'data');
        view.setUint32(40, dataLength, true);

        // Write 16-bit PCM samples
        let offset = 44;
        for (let i = 0; i < samples.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            // Convert to 16-bit signed int (-32768 to 32767)
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    _writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /** Subtle Single-Line Waveform Visualizer */
    _startWaveformAnimation() {
        const canvas = this.visualizerCanvas;
        const ctx = canvas.getContext('2d');
        let phase = 0;

        const render = () => {
            if (!this.recording) return;

            const width = canvas.width;
            const height = canvas.height;
            ctx.clearRect(0, 0, width, height);

            const midY = height / 2;
            const amplitude = Math.max(3, Math.min(height * 0.42, this.currentRms * height * 2.8));

            ctx.beginPath();
            ctx.lineWidth = 2.0;
            ctx.strokeStyle = '#6366F1'; // Pastel Indigo Accent
            ctx.lineCap = 'round';

            for (let x = 0; x < width; x++) {
                const normalizedX = x / width;
                // Soft sine wave envelope modulated by RMS energy
                const envelope = Math.sin(normalizedX * Math.PI);
                const y = midY + Math.sin(x * 0.05 + phase) * amplitude * envelope;

                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            phase += 0.15;
            this.animationId = requestAnimationFrame(render);
        };

        render();
    }

    _clearWaveformCanvas() {
        if (!this.visualizerCanvas) return;
        const canvas = this.visualizerCanvas;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw calm flat center line
        ctx.beginPath();
        ctx.strokeStyle = '#E2E8F0';
        ctx.lineWidth = 1;
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
    }
}

window.AudioRecorder = AudioRecorder;
