
import { PCMPlayerWorklet as PCMPlayer } from './PCMPlayerWorklet.js';

// Configuration
const SAMPLE_RATE = 24000;
const FADE_SAMPLES = 480; // 20ms fade at 24kHz
const REALTIME_THRESHOLD = 1.0;

export class PocketTTSStreaming {
    constructor() {
        this.worker = null;
        this.player = null;
        this.audioContext = null;
        this.isGenerating = false;
        this.isWorkerReady = false;
        this.isVoicePreparing = false;
        this.pendingGeneration = false;

        // Voice state
        this.availableVoices = [];
        this.currentVoice = null;
        this.customVoiceAudio = null;

        // Metrics State
        this.generationStartTime = 0;
        this.lastChunkFinishTime = 0;
        this.rtfMovingAverage = 0;
        this.skipNextRtf = false;

        // Edge optimization state (dynamic LSD)
        this.edgeOptimizationApplied = false;
        this.playbackMode = 'pending'; // pending | stream | buffer_all
        this.bufferedChunks = [];
        this.deferStreamEnd = false;

        this.elements = {
            textInput: document.getElementById('text-input'),
            generateBtn: document.getElementById('generate-btn'),
            stopBtn: document.getElementById('stop-btn'),
            statusText: document.getElementById('stat-status'),
            statusIndicator: document.getElementById('status-indicator'),
            modelStatusIcon: document.querySelector('#model-status .model-status__dot'),
            modelStatusText: document.querySelector('#model-status .model-status__text'),
            btnLoader: document.getElementById('btn-loader'),
            statTTFB: document.getElementById('stat-ttfb'),
            statRTFx: document.getElementById('stat-rtfx'),
            ttfbBar: document.getElementById('ttfb-bar'),
            rtfxContext: document.getElementById('rtfx-context'),
            edgeOptNote: document.getElementById('edge-opt-note'),
            fullGenNote: document.getElementById('full-gen-note'),
            voiceSelect: document.getElementById('voice-select'),
            voiceUpload: document.getElementById('voice-upload'),
            voiceUploadBtn: document.getElementById('voice-upload-btn'),
            voiceUploadStatus: document.getElementById('voice-upload-status')
        };

        this.attachEventListeners();
        this.init();
        this.setupVisualization();
    }

    async init() {
        console.log('Pocket TTS v1.0 - Web Demo');
        console.log('Secure context:', window.isSecureContext);
        console.log('Location:', window.location.href);
        this.updateStatus('Initializing...', 'running');

        // Initial button state
        this.elements.generateBtn.disabled = true;
        if (this.elements.voiceUploadBtn) this.elements.voiceUploadBtn.disabled = true;
        if (this.elements.voiceSelect) this.elements.voiceSelect.disabled = true;
        const btnText = this.elements.generateBtn.querySelector('.btn__text');
        if (btnText) btnText.textContent = 'Loading Models...';
        this.elements.btnLoader.style.display = 'block';

        // Check secure context
        if (!window.isSecureContext) {
            const msg = 'AudioWorklet requires HTTPS or localhost. Current: ' + window.location.hostname;
            console.error(msg);
            this.updateStatus(msg, 'error');
            this.elements.btnLoader.style.display = 'none';
            if (btnText) btnText.textContent = 'Secure Context Required';
            return;
        }

        try {
            // Initialize Audio Context and Player
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE,
                latencyHint: 'interactive'
            });

            // Check if AudioWorklet is supported
            if (!this.audioContext.audioWorklet) {
                throw new Error('AudioWorklet not supported in this browser.');
            }

            await this.audioContext.audioWorklet.addModule('PCMPlayerWorklet.js');
            this.player = new PCMPlayer(this.audioContext);
            this.player.addEventListener('audioEnded', () => {
                if (this.deferStreamEnd) {
                    this.deferStreamEnd = false;
                    this.finalizePlayback();
                }
            });
        } catch (err) {
            console.error('Audio initialization failed:', err);
            this.updateStatus('Audio init failed: ' + err.message, 'error');
            this.elements.btnLoader.style.display = 'none';
            if (btnText) btnText.textContent = 'Audio Error';
            return;
        }

        // Initialize Worker (as ES module)
        console.log('Spawning Inference Worker...');
        this.worker = new Worker('./inference-worker.js?v=15', { type: 'module' });

        this.worker.onmessage = (e) => {
            const { type, data, error, status, state, metrics, text, voices, defaultVoice, voiceName } = e.data;

            switch (type) {
                case 'status':
                    this.updateStatus(status, state);
                    break;
                case 'model_status':
                    this.updateModelStatus(status, text);
                    break;
                case 'voices_loaded':
                    this.handleVoicesLoaded(voices, defaultVoice);
                    break;
                case 'voice_encoded':
                    this.handleVoiceEncoded(voiceName);
                    this.finishVoicePreparation();
                    break;
                case 'voice_set':
                    this.currentVoice = voiceName;
                    this.finishVoicePreparation();
                    break;
                case 'loaded':
                    console.log('Worker confirmed models loaded.');
                    this.isWorkerReady = true;
                    this.resetUI();

                    if (this.pendingGeneration) {
                        this.pendingGeneration = false;
                        this.startGeneration();
                    }
                    break;
                case 'generation_started':
                    // The main thread already sets this in startGeneration for better precision
                    break;
                case 'audio_chunk':
                    this.handleAudioChunk(data, metrics);
                    break;
                case 'stream_ended':
                    this.handleStreamEnd();
                    break;
                case 'error':
                    console.error('Worker Error:', error);
                    this.updateStatus(`Error: ${error}`, 'error');
                    this.finishVoicePreparation();
                    this.resetUI();
                    break;
            }
        };

        // Trigger Model Load in Worker
        this.worker.postMessage({ type: 'load' });
    }

    handleVoicesLoaded(voices, defaultVoice) {
        this.availableVoices = voices;
        this.currentVoice = defaultVoice;

        // Populate voice selector
        if (this.elements.voiceSelect) {
            this.elements.voiceSelect.innerHTML = '';

            // Add predefined voices
            for (const voice of voices) {
                const option = document.createElement('option');
                option.value = voice;
                option.textContent = voice.charAt(0).toUpperCase() + voice.slice(1);
                if (voice === defaultVoice) {
                    option.selected = true;
                }
                this.elements.voiceSelect.appendChild(option);
            }

            // Add custom voice option
            const customOption = document.createElement('option');
            customOption.value = 'custom';
            customOption.textContent = 'Custom (Upload)';
            this.elements.voiceSelect.appendChild(customOption);
        }

        console.log('Available voices:', voices, 'Default:', defaultVoice);
    }

    startVoicePreparation(statusText = 'Preparing voice...') {
        this.isVoicePreparing = true;
        this.elements.generateBtn.disabled = true;
        const btnText = this.elements.generateBtn.querySelector('.btn__text');
        if (btnText) btnText.textContent = 'Preparing Voice...';
        this.elements.btnLoader.style.display = 'block';
        if (this.elements.voiceUploadBtn) this.elements.voiceUploadBtn.disabled = true;
        if (this.elements.voiceSelect) this.elements.voiceSelect.disabled = true;
        this.updateStatus(statusText, 'loading');
    }

    finishVoicePreparation() {
        this.isVoicePreparing = false;
        if (!this.isWorkerReady || this.isGenerating) return;
        this.resetUI();
    }

    handleVoiceEncoded(voiceName) {
        this.currentVoice = voiceName;
        if (this.elements.voiceUploadStatus) {
            this.elements.voiceUploadStatus.textContent = 'Voice encoded successfully!';
            this.elements.voiceUploadStatus.className = 'voice-upload-status success';
        }
        // Set the select to custom
        if (this.elements.voiceSelect) {
            this.elements.voiceSelect.value = 'custom';
        }
    }

    async handleVoiceUpload(file) {
        if (!file) return;
        this.startVoicePreparation('Preparing custom voice...');

        if (this.elements.voiceUploadStatus) {
            this.elements.voiceUploadStatus.textContent = 'Processing audio...';
            this.elements.voiceUploadStatus.className = 'voice-upload-status';
        }

        try {
            // Decode audio file
            const arrayBuffer = await file.arrayBuffer();
            const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

            // Resample to 24kHz if needed
            let audioData;
            if (audioBuffer.sampleRate !== SAMPLE_RATE) {
                audioData = this.resampleAudio(audioBuffer, SAMPLE_RATE);
            } else {
                audioData = audioBuffer.getChannelData(0);
            }

            // Convert to mono if stereo
            if (audioBuffer.numberOfChannels > 1 && audioBuffer.sampleRate === SAMPLE_RATE) {
                const left = audioBuffer.getChannelData(0);
                const right = audioBuffer.getChannelData(1);
                audioData = new Float32Array(left.length);
                for (let i = 0; i < left.length; i++) {
                    audioData[i] = (left[i] + right[i]) / 2;
                }
            }

            // Limit to 10 seconds max
            const maxSamples = SAMPLE_RATE * 10;
            if (audioData.length > maxSamples) {
                audioData = audioData.slice(0, maxSamples);
            }

            // Send to worker for encoding
            this.worker.postMessage({
                type: 'encode_voice',
                data: { audio: audioData }
            });

        } catch (err) {
            console.error('Voice upload error:', err);
            this.finishVoicePreparation();
            if (this.elements.voiceUploadStatus) {
                this.elements.voiceUploadStatus.textContent = `Error: ${err.message}`;
                this.elements.voiceUploadStatus.className = 'voice-upload-status error';
            }
        }
    }

    resampleAudio(audioBuffer, targetRate) {
        const sourceRate = audioBuffer.sampleRate;
        const sourceData = audioBuffer.getChannelData(0);

        // If stereo, mix to mono
        let monoData = sourceData;
        if (audioBuffer.numberOfChannels > 1) {
            const right = audioBuffer.getChannelData(1);
            monoData = new Float32Array(sourceData.length);
            for (let i = 0; i < sourceData.length; i++) {
                monoData[i] = (sourceData[i] + right[i]) / 2;
            }
        }

        // Linear interpolation resampling
        const ratio = sourceRate / targetRate;
        const outputLength = Math.floor(monoData.length / ratio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, monoData.length - 1);
            const t = srcIndex - srcIndexFloor;
            output[i] = monoData[srcIndexFloor] * (1 - t) + monoData[srcIndexCeil] * t;
        }

        return output;
    }

    attachEventListeners() {
        this.elements.generateBtn.addEventListener('click', () => this.startGeneration());
        this.elements.stopBtn.addEventListener('click', () => this.stopGeneration());

        // Voice selector
        if (this.elements.voiceSelect) {
            this.elements.voiceSelect.addEventListener('change', (e) => {
                const voice = e.target.value;
                if (voice === 'custom') {
                    // Trigger file upload
                    if (this.elements.voiceUpload) {
                        this.elements.voiceUpload.click();
                    }
                } else {
                    this.startVoicePreparation(`Switching to ${voice} voice...`);
                    this.worker.postMessage({
                        type: 'set_voice',
                        data: { voiceName: voice }
                    });
                }
            });
        }

        // Voice file upload
        if (this.elements.voiceUpload) {
            this.elements.voiceUpload.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    this.handleVoiceUpload(file);
                }
            });
        }

        // Voice upload button
        if (this.elements.voiceUploadBtn) {
            this.elements.voiceUploadBtn.addEventListener('click', () => {
                if (this.elements.voiceUpload) {
                    this.elements.voiceUpload.click();
                }
            });
        }

        // Sample buttons
        document.querySelectorAll('.sample-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.elements.textInput.value = btn.getAttribute('data-text');
                // Trigger input event to update character count
                this.elements.textInput.dispatchEvent(new Event('input'));
            });
        });

        // Character count
        this.elements.textInput.addEventListener('input', () => {
            const count = this.elements.textInput.value.length;
            const countEl = document.getElementById('char-count');
            if (countEl) countEl.textContent = count;
        });

        this.elements.textInput.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                this.startGeneration();
            }
        });
    }

    async startGeneration() {
        this.generationStartTime = performance.now();
        try {
            if (!this.isWorkerReady) {
                this.pendingGeneration = true;
                const btnText = this.elements.generateBtn.querySelector('.btn__text');
                if (btnText) btnText.textContent = 'Starting soon...';
                return;
            }
            if (this.isVoicePreparing) {
                return;
            }

            if (this.isGenerating) return;

            if (this.audioContext && this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            const text = this.elements.textInput.value.trim();
            if (!text) return;

            this.isGenerating = true;
            this.elements.generateBtn.disabled = true;
            this.elements.generateBtn.classList.add('btn--generating');
            this.elements.stopBtn.disabled = false;

            if (this.player) this.player.reset();

            // Reset metrics
            this.elements.statTTFB.textContent = '--';
            this.elements.statRTFx.textContent = '--';
            if (this.elements.ttfbBar) this.elements.ttfbBar.style.width = '0%';
            if (this.elements.edgeOptNote) this.elements.edgeOptNote.style.display = 'none';
            if (this.elements.fullGenNote) this.elements.fullGenNote.style.display = 'none';

            this.rtfMovingAverage = 0;
            this.edgeOptimizationApplied = false;
            this.lastChunkFinishTime = 0;
            this.skipNextRtf = false;
            this.playbackMode = 'pending';
            this.bufferedChunks = [];
            this.deferStreamEnd = false;

            // Get current voice from selector
            const voice = this.elements.voiceSelect ? this.elements.voiceSelect.value : this.currentVoice;

            this.worker.postMessage({
                type: 'generate',
                data: { text, voice }
            });
        } catch (err) {
            console.error('Error in startGeneration:', err);
            this.updateStatus(`Error: ${err.message}`, 'error');
            this.isGenerating = false;
            this.resetUI();
        }
    }

    stopGeneration() {
        if (!this.isGenerating) return;
        this.worker.postMessage({ type: 'stop' });
        // Handle stop immediately in UI
        this.handleStreamEnd();
    }

    applyFadeIn(audioData) {
        const fadeLen = Math.min(FADE_SAMPLES, audioData.length);
        for (let i = 0; i < fadeLen; i++) {
            audioData[i] *= i / fadeLen;
        }
    }

    applyFadeOut(audioData) {
        const fadeLen = Math.min(FADE_SAMPLES, audioData.length);
        const startIdx = audioData.length - fadeLen;
        for (let i = 0; i < fadeLen; i++) {
            audioData[startIdx + i] *= 1 - (i / fadeLen);
        }
    }

    bufferOrPlay(audioData) {
        if (this.playbackMode === 'stream') {
            this.player.playAudio(audioData);
        } else {
            this.bufferedChunks.push(audioData);
        }
    }

    flushBufferedAudio() {
        if (!this.bufferedChunks.length) return;
        for (const chunk of this.bufferedChunks) {
            this.player.playAudio(chunk);
        }
        this.bufferedChunks = [];
    }

    switchToStream() {
        this.playbackMode = 'stream';
        if (this.elements.fullGenNote) this.elements.fullGenNote.style.display = 'none';
        this.flushBufferedAudio();
    }

    switchToBufferAll() {
        this.playbackMode = 'buffer_all';
        if (this.elements.fullGenNote) this.elements.fullGenNote.style.display = 'block';
    }

    finalizePlayback() {
        this.isGenerating = false;
        this.resetUI();
        if (this.elements.fullGenNote) this.elements.fullGenNote.style.display = 'none';
    }

    handleAudioChunk(audioData, metrics) {
        if (!this.isGenerating) return;

        if (metrics.isSilence) {
            this.bufferOrPlay(audioData);
            this.skipNextRtf = true;
            return;
        }

        // Apply fade-in at the start of each text chunk
        if (metrics.isFirst || metrics.chunkStart) this.applyFadeIn(audioData);
        if (metrics.isLast) this.applyFadeOut(audioData);

        // Calculate RTFx immediately (not in RAF) so edge optimization triggers fast
        const now = performance.now();
        let ttfb = 0;
        let instantaneousRTF = 0;
        let arrivalRTF = 0;

        if (metrics.isFirst) {
            ttfb = now - this.generationStartTime;
            this.lastChunkFinishTime = now;
        } else if (this.skipNextRtf) {
            this.lastChunkFinishTime = now;
            this.skipNextRtf = false;
        } else if (this.lastChunkFinishTime > 0) {
            const timeSinceLastChunk = (now - this.lastChunkFinishTime) / 1000;
            this.lastChunkFinishTime = now;

            if (timeSinceLastChunk > 0) {
                arrivalRTF = metrics.chunkDuration / timeSinceLastChunk;
            }
        }

        // Prefer actual generation time when available (avoids TTFB skew)
        if (metrics.genTimeSec && metrics.genTimeSec > 0) {
            instantaneousRTF = metrics.chunkDuration / metrics.genTimeSec;
        } else if (arrivalRTF > 0) {
            instantaneousRTF = arrivalRTF;
        }

        if (instantaneousRTF > 0) {
            if (this.rtfMovingAverage === 0) {
                this.rtfMovingAverage = instantaneousRTF;
            } else {
                this.rtfMovingAverage = this.rtfMovingAverage * 0.8 + instantaneousRTF * 0.2;
            }

            const edgeRtf = arrivalRTF > 0 ? arrivalRTF : instantaneousRTF;
            if (!this.edgeOptimizationApplied && edgeRtf < REALTIME_THRESHOLD) {
                this.edgeOptimizationApplied = true;
                this.worker.postMessage({ type: 'set_lsd', data: { lsd: 1 } });
                console.log('Edge optimization applied: LSD reduced to 1');
            }
        }

        if (this.playbackMode === 'pending') {
            if (instantaneousRTF >= REALTIME_THRESHOLD) {
                this.switchToStream();
            } else if (!metrics.isFirst && this.edgeOptimizationApplied && instantaneousRTF < REALTIME_THRESHOLD) {
                this.switchToBufferAll();
            }
        }

        this.bufferOrPlay(audioData);

        // Update UI in RAF (non-blocking)
        const rtfxToDisplay = this.rtfMovingAverage;
        const showEdgeOpt = this.edgeOptimizationApplied;
        requestAnimationFrame(() => {
            if (metrics.isFirst) {
                this.updateTTFB(ttfb);
            }
            if (rtfxToDisplay > 0) {
                this.updateRTFx(rtfxToDisplay);
            }
            if (showEdgeOpt && this.elements.edgeOptNote) {
                this.elements.edgeOptNote.style.display = 'block';
            }
        });
    }

    handleStreamEnd() {
        if (this.playbackMode === 'pending') {
            this.switchToBufferAll();
        }

        if (this.playbackMode === 'buffer_all') {
            this.flushBufferedAudio();
            if (this.player.notifyStreamEnded) this.player.notifyStreamEnded();
            this.deferStreamEnd = true;
            return;
        }

        if (this.player.notifyStreamEnded) this.player.notifyStreamEnded();
        this.finalizePlayback();
    }

    resetUI() {
        const canGenerate = this.isWorkerReady && !this.isVoicePreparing && !this.isGenerating;
        this.elements.generateBtn.disabled = !canGenerate;
        this.elements.generateBtn.classList.remove('btn--generating');
        const btnText = this.elements.generateBtn.querySelector('.btn__text');
        if (btnText) {
            if (!this.isWorkerReady) btnText.textContent = 'Loading Models...';
            else if (this.isVoicePreparing) btnText.textContent = 'Preparing Voice...';
            else btnText.textContent = 'Generate Audio';
        }
        this.elements.stopBtn.disabled = true;
        if (this.elements.voiceUploadBtn) this.elements.voiceUploadBtn.disabled = !canGenerate;
        if (this.elements.voiceSelect) this.elements.voiceSelect.disabled = !canGenerate;
        this.elements.btnLoader.style.display = canGenerate ? 'none' : 'block';
    }

    updateStatus(text, state) {
        this.elements.statusText.textContent = text;
        this.elements.statusIndicator.className = `status-indicator status-${state}`;
    }

    updateModelStatus(state, text) {
        this.elements.modelStatusText.textContent = text;
        this.elements.modelStatusIcon.className = `status-icon status-${state}`;
    }

    updateTTFB(ms) {
        this.elements.statTTFB.textContent = Math.round(ms);
        const percentage = Math.min((ms / 2000) * 100, 100);
        this.elements.ttfbBar.style.width = `${percentage}%`;
        this.elements.ttfbBar.style.background = ms < 500 ? '#00d4aa' : ms < 1000 ? '#ffd93d' : '#ff6b6b';
    }

    updateRTFx(val) {
        this.elements.statRTFx.textContent = `${val.toFixed(2)}x`;
        this.elements.rtfxContext.style.color = val >= 1.0 ? '#00d4aa' : '#ff6b6b';
    }

    // -------------------------------------------------------------------------
    // Visualization
    // -------------------------------------------------------------------------
    setupVisualization() {
        this.waveformCanvas = document.getElementById('visualizer-waveform');
        this.barsCanvas = document.getElementById('visualizer-bars');
        if (!this.waveformCanvas || !this.barsCanvas) return;

        this.waveformCtx = this.waveformCanvas.getContext('2d');
        this.barsCtx = this.barsCanvas.getContext('2d');

        // Initial resize
        this.resizeCanvases();
        window.addEventListener('resize', () => this.resizeCanvases());

        // Start animation loop
        requestAnimationFrame(() => this.draw());
    }

    resizeCanvases() {
        if (!this.waveformCanvas || !this.barsCanvas) return;

        const parent = this.waveformCanvas.parentElement;
        const width = parent.clientWidth;
        const height = parent.clientHeight;

        const dpr = window.devicePixelRatio || 1;

        [this.waveformCanvas, this.barsCanvas].forEach(canvas => {
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);
        });
    }

    draw() {
        requestAnimationFrame(() => this.draw());

        if (!this.player || !this.player.analyser) return;

        const bufferLength = this.player.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        // Draw Bars (Frequency)
        this.player.analyser.getByteFrequencyData(dataArray);
        this.drawBars(dataArray);

        // Draw Waveform (Time Domain)
        this.player.analyser.getByteTimeDomainData(dataArray);
        this.drawWaveform(dataArray);
    }

    drawWaveform(dataArray) {
        const ctx = this.waveformCtx;
        const canvas = this.waveformCanvas;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.clearRect(0, 0, width, height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00d4aa'; // Mint primary
        ctx.beginPath();

        const sliceWidth = width / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * height) / 2;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);

            x += sliceWidth;
        }

        ctx.lineTo(width, height / 2);
        ctx.stroke();
    }

    drawBars(dataArray) {
        const ctx = this.barsCtx;
        const canvas = this.barsCanvas;
        const width = canvas.width / (window.devicePixelRatio || 1);
        const height = canvas.height / (window.devicePixelRatio || 1);

        ctx.clearRect(0, 0, width, height);

        const barCount = 120; // Number of bars to display
        const barWidth = (width / barCount);
        const samplesPerBar = Math.floor(dataArray.length / barCount);

        for (let i = 0; i < barCount; i++) {
            let sum = 0;
            for (let j = 0; j < samplesPerBar; j++) {
                sum += dataArray[i * samplesPerBar + j];
            }
            const average = sum / samplesPerBar;
            const barHeight = (average / 255) * height * 0.8;

            // Gradient for bar - Mint spectrum
            const gradient = ctx.createLinearGradient(0, height, 0, height - barHeight);
            gradient.addColorStop(0, '#3eb48944');
            gradient.addColorStop(1, '#7fffd4cc');

            ctx.fillStyle = gradient;

            // Rounded bars
            const x = i * barWidth;
            const y = height - barHeight;
            const radius = barWidth / 2;

            ctx.beginPath();
            ctx.roundRect(x + 1, y, barWidth - 2, barHeight, [2, 2, 0, 0]);
            ctx.fill();
        }
    }
}

// Start the app
document.addEventListener('DOMContentLoaded', () => {
    window.app = new PocketTTSStreaming();
});
