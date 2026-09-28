/**
 * VoiceGate SPA Controller.
 * Interacts with VoiceGate FastAPI backend.
 */

document.addEventListener('DOMContentLoaded', async () => {
    const api = new ApiClient('');
    const recorder = new AudioRecorder();
    const sounds = new SoundEffects();

    // -------------------------------------------------------------
    // App State
    // -------------------------------------------------------------
    const state = {
        identities: [],
        isRecording: false,
        recordingType: null, // 'enroll' | 'verify'
        sessionId: localStorage.getItem('voicegate_session_id'),
    };

    // -------------------------------------------------------------
    // Session Init
    // -------------------------------------------------------------
    async function initSession() {
        try {
            if (!state.sessionId) {
                const res = await api.createSession();
                state.sessionId = res.session_id;
                localStorage.setItem('voicegate_session_id', state.sessionId);
            }
            await refreshIdentities();
        } catch (err) {
            console.error("Session init failed", err);
            // Fallback to DEFAULT
            state.sessionId = 'DEFAULT';
            localStorage.setItem('voicegate_session_id', 'DEFAULT');
            await refreshIdentities();
        }
    }

    // -------------------------------------------------------------
    // Suggested Speech Prompts
    // -------------------------------------------------------------
    const SUGGESTED_PHRASES = [
        "My unique voiceprint serves as my biometric passport today, allowing the neural network to analyze my natural tone.",
        "Artificial intelligence listens closely to every harmonic frequency in human speech to distinguish between genuine users and clever impostors.",
        "Sphinx of black quartz, please judge my vocal vow as I speak clearly into the microphone so the machine learns my voice.",
        "To travel across galaxies and explore uncharted stars, we must first master the art of clear communication.",
        "Every great mystery begins with a quiet whisper in the shadows, waiting for someone perceptive enough to uncover the hidden truth.",
        "When the cool autumn wind sweeps through the golden valley, old memories return of warm campfires, laughter, and stories shared."
    ];
    let phraseIndex = 0;

    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    const enrolledCountEl = document.getElementById('enrolled-count');
    const btnToggleSound = document.getElementById('btn-toggle-sound');
    const btnOpenGuide = document.getElementById('btn-open-guide');
    const btnOpenReset = document.getElementById('btn-open-reset');

    const inputPlayerName = document.getElementById('input-player-name');
    const btnRecordReg = document.getElementById('btn-record-reg');
    const textRecordReg = document.getElementById('text-record-reg');
    const textSuggestedPhrase = document.getElementById('text-suggested-phrase');
    const btnShufflePhrase = document.getElementById('btn-shuffle-phrase');
    const rosterEmptyEl = document.getElementById('roster-empty');
    const rosterChipsEl = document.getElementById('roster-chips');

    const selectVerifyTarget = document.getElementById('select-verify-target');
    const btnRecordVerify = document.getElementById('btn-record-verify');
    const textRecordVerify = document.getElementById('text-record-verify');
    const canvasVerify = document.getElementById('canvas-verify-waveform');
    
    const cardVerifyResult = document.getElementById('card-verify-result');
    const textVerdict = document.getElementById('text-verdict');
    const textMatchConfidence = document.getElementById('text-match-confidence');
    const textMatchMargin = document.getElementById('text-match-margin');
    const verifyRankingsList = document.getElementById('verify-rankings-list');

    const modalGuide = document.getElementById('modal-guide');
    const btnCloseGuide = document.getElementById('btn-close-guide');
    const btnGuideGotIt = document.getElementById('btn-guide-got-it');

    const modalReset = document.getElementById('modal-reset');
    const btnCloseReset = document.getElementById('btn-close-reset');
    const btnConfirmResetAll = document.getElementById('btn-confirm-reset-all');

    const toastContainer = document.getElementById('toast-container');

    // -------------------------------------------------------------
    // Toast Notification
    // -------------------------------------------------------------
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast';
        toast.innerText = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.2s ease';
            setTimeout(() => toast.remove(), 200);
        }, 2800);
    }

    // -------------------------------------------------------------
    // Sound FX Toggle
    // -------------------------------------------------------------
    function updateSoundBtn() {
        if (sounds.isEnabled()) {
            btnToggleSound.classList.add('active');
        } else {
            btnToggleSound.classList.remove('active');
        }
    }

    btnToggleSound.addEventListener('click', () => {
        const on = sounds.toggle();
        updateSoundBtn();
        showToast(on ? 'Sound on' : 'Sound muted');
    });
    updateSoundBtn();

    // -------------------------------------------------------------
    // Suggested Speech Phrase Shuffle
    // -------------------------------------------------------------
    btnShufflePhrase.addEventListener('click', () => {
        sounds.click();
        phraseIndex = (phraseIndex + 1) % SUGGESTED_PHRASES.length;
        textSuggestedPhrase.innerText = `"${SUGGESTED_PHRASES[phraseIndex]}"`;
    });

    // -------------------------------------------------------------
    // Identities Management
    // -------------------------------------------------------------
    async function refreshIdentities() {
        try {
            const res = await api.getIdentities(state.sessionId);
            state.identities = res.identities;
            renderIdentities();
            updateTargetDropdown();
            enrolledCountEl.innerText = state.identities.length === 1 ? '1 Identity' : `${state.identities.length} Identities`;
        } catch (err) {
            console.error(err);
            showToast("Failed to fetch identities");
        }
    }

    function renderIdentities() {
        rosterChipsEl.innerHTML = '';
        if (state.identities.length === 0) {
            rosterEmptyEl.classList.remove('hidden');
            rosterChipsEl.classList.add('hidden');
            return;
        }

        rosterEmptyEl.classList.add('hidden');
        rosterChipsEl.classList.remove('hidden');

        state.identities.forEach((p) => {
            const chip = document.createElement('div');
            chip.className = 'player-chip';

            const name = document.createElement('span');
            name.className = 'player-chip-name';
            name.innerText = p.name;

            const del = document.createElement('button');
            del.className = 'player-chip-del';
            del.title = `Delete ${p.name}`;
            del.innerHTML = '&times;';
            del.addEventListener('click', async (e) => {
                e.stopPropagation();
                sounds.click();
                try {
                    await api.deleteIdentity(p.identity_id, state.sessionId);
                    showToast(`Removed '${p.name}'`);
                    await refreshIdentities();
                } catch (err) {
                    showToast("Failed to delete identity");
                }
            });

            chip.appendChild(name);
            chip.appendChild(del);
            rosterChipsEl.appendChild(chip);
        });
    }

    function updateTargetDropdown() {
        const val = selectVerifyTarget.value;
        selectVerifyTarget.innerHTML = '<option value="">Any Identity (1:N)</option>';
        state.identities.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = p.identity_id;
            opt.innerText = p.name;
            selectVerifyTarget.appendChild(opt);
        });
        if (val && state.identities.some((p) => p.identity_id === val)) {
            selectVerifyTarget.value = val;
        }
    }

    // -------------------------------------------------------------
    // Recording: Enroll Identity
    // -------------------------------------------------------------
    let enrollTimer = null;

    btnRecordReg.addEventListener('click', async () => {
        const name = inputPlayerName.value.trim();
        if (!name) {
            showToast('Enter name first');
            inputPlayerName.focus();
            return;
        }

        if (state.isRecording) {
            await finishEnrollment(name);
            return;
        }

        try {
            await recorder.start();
            sounds.recordStart();
            state.isRecording = true;
            state.recordingType = 'enroll';

            btnRecordReg.classList.add('recording');
            textRecordReg.innerText = 'Recording (10s)...';

            let timeLeft = 10;
            enrollTimer = setInterval(async () => {
                timeLeft--;
                if (timeLeft > 0) {
                    sounds.tick();
                    textRecordReg.innerText = `Recording (${timeLeft}s)...`;
                } else {
                    clearInterval(enrollTimer);
                    await finishEnrollment(name);
                }
            }, 1000);
        } catch (err) {
            showToast('Microphone access denied');
        }
    });

    async function finishEnrollment(name) {
        if (enrollTimer) {
            clearInterval(enrollTimer);
            enrollTimer = null;
        }

        sounds.recordStop();
        textRecordReg.innerText = 'Processing...';
        btnRecordReg.classList.remove('recording');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;
            if (!blob) return;

            await api.enrollIdentity(name, blob, state.sessionId);

            sounds.successChime();
            showToast(`'${name}' enrolled!`);
            inputPlayerName.value = '';
            await refreshIdentities();
        } catch (err) {
            sounds.failureTone();
            showToast(err.message || 'Enrollment failed');
        } finally {
            textRecordReg.innerText = 'Record Sample';
        }
    }

    // -------------------------------------------------------------
    // Recording: Verify Speaker
    // -------------------------------------------------------------
    btnRecordVerify.addEventListener('click', async () => {
        if (state.identities.length === 0) {
            showToast('Enroll at least 1 identity first');
            return;
        }

        if (state.isRecording) {
            await finishVerify();
            return;
        }

        try {
            await recorder.start(canvasVerify);
            sounds.recordStart();
            state.isRecording = true;
            state.recordingType = 'verify';

            cardVerifyResult.classList.add('hidden');
            btnRecordVerify.classList.add('recording');
            textRecordVerify.innerText = 'Listening... (Tap to stop)';
        } catch (err) {
            showToast('Microphone access denied');
        }
    });

    async function finishVerify() {
        sounds.recordStop();
        textRecordVerify.innerText = 'Analyzing...';
        btnRecordVerify.classList.remove('recording');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;
            if (!blob) return;

            const targetId = selectVerifyTarget.value || null;
            const res = await api.verifySpeaker(blob, state.sessionId, targetId);

            textVerdict.innerText = res.verdict;
            if (res.verdict === "MATCH") {
                textVerdict.style.color = "var(--emerald)";
                sounds.successChime();
            } else if (res.verdict === "UNCERTAIN") {
                textVerdict.style.color = "var(--amber)";
                sounds.failureTone();
            } else {
                textVerdict.style.color = "var(--rose)";
                sounds.failureTone();
            }

            textMatchConfidence.innerText = `${res.confidence_percent}% Score`;
            textMatchMargin.innerText = res.is_certain ? 'High Margin' : 'Disputed / Low Margin';

            // Candidate breakdown
            verifyRankingsList.innerHTML = '';
            if (res.rankings && res.rankings.length > 0) {
                res.rankings.forEach((cand) => {
                    const row = document.createElement('div');
                    row.className = 'ranking-row';

                    const name = document.createElement('span');
                    name.innerText = cand.name;

                    const pct = document.createElement('span');
                    pct.style.fontWeight = '600';
                    pct.style.color = cand.identity_id === res.identity_id ? 'var(--accent)' : 'var(--text-muted)';
                    pct.innerText = cand.probability_percent !== undefined
                        ? `${cand.similarity_percent}% (${cand.probability_percent}% prob)`
                        : `${cand.similarity_percent}%`;

                    row.appendChild(name);
                    row.appendChild(pct);
                    verifyRankingsList.appendChild(row);
                });
            }

            cardVerifyResult.classList.remove('hidden');
        } catch (err) {
            sounds.failureTone();
            showToast(err.message || 'Verification failed');
        } finally {
            textRecordVerify.innerText = 'Verify Identity';
        }
    }

    // -------------------------------------------------------------
    // Modals: Guide, Reset
    // -------------------------------------------------------------
    btnOpenGuide.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.remove('hidden');
    });

    btnCloseGuide.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.add('hidden');
    });

    btnGuideGotIt.addEventListener('click', () => {
        sounds.click();
        modalGuide.classList.add('hidden');
    });

    btnOpenReset.addEventListener('click', () => {
        sounds.click();
        modalReset.classList.remove('hidden');
    });

    btnCloseReset.addEventListener('click', () => {
        sounds.click();
        modalReset.classList.add('hidden');
    });

    btnConfirmResetAll.addEventListener('click', async () => {
        sounds.click();
        try {
            // Re-create a new session
            const res = await api.createSession();
            state.sessionId = res.session_id;
            localStorage.setItem('voicegate_session_id', state.sessionId);
            showToast('Session cleared');
            modalReset.classList.add('hidden');
            cardVerifyResult.classList.add('hidden');
            await refreshIdentities();
        } catch (err) {
            showToast("Failed to clear session");
        }
    });

    // Esc closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            modalGuide.classList.add('hidden');
            modalReset.classList.add('hidden');
        }
    });

    // Start
    await initSession();
});
