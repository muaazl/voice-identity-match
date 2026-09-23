/**
 * VoiceMimic SPA Controller & State Machine.
 * Manages UI transitions, recording triggers, game flow, and modals.
 */

document.addEventListener('DOMContentLoaded', () => {
    const api = new ApiClient();
    const recorder = new AudioRecorder();

    // -------------------------------------------------------------
    // App State
    // -------------------------------------------------------------
    let state = {
        players: [],
        activeMode: 'guess_who', // 'guess_who' | 'impostor'
        isRecording: false,
        recordingType: null, // 'reg' | 'guess' | 'mimic'
        currentRound: null, // stores last guess-who round
        impostorTargetId: null,
    };

    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    // Header & Stats
    const enrolledCountEl = document.getElementById('enrolled-count');
    const btnOpenLeaderboard = document.getElementById('btn-open-leaderboard');
    const btnOpenReset = document.getElementById('btn-open-reset');

    // Mode Toggle Tabs
    const tabGuessWho = document.getElementById('tab-guess-who');
    const tabImpostor = document.getElementById('tab-impostor');
    const sectionGuessWho = document.getElementById('section-guess-who');
    const sectionImpostor = document.getElementById('section-impostor');

    // Registration Card
    const inputPlayerName = document.getElementById('input-player-name');
    const btnRecordReg = document.getElementById('btn-record-reg');
    const textRecordReg = document.getElementById('text-record-reg');
    const rosterListEl = document.getElementById('roster-list');
    const rosterEmptyEl = document.getElementById('roster-empty');

    // Arena: Guess Who
    const btnRecordGuess = document.getElementById('btn-record-guess');
    const textRecordGuess = document.getElementById('text-record-guess');
    const canvasGuess = document.getElementById('canvas-guess-waveform');
    const cardGuessResult = document.getElementById('card-guess-result');
    const textPredictedName = document.getElementById('text-predicted-name');
    const textMatchConfidence = document.getElementById('text-match-confidence');
    const textMatchMargin = document.getElementById('text-match-margin');
    const btnGuessCorrect = document.getElementById('btn-guess-correct');
    const btnGuessWrong = document.getElementById('btn-guess-wrong');

    // Arena: Impostor Challenge
    const selectImpostorTarget = document.getElementById('select-impostor-target');
    const btnRecordMimic = document.getElementById('btn-record-mimic');
    const textRecordMimic = document.getElementById('text-record-mimic');
    const canvasMimic = document.getElementById('canvas-mimic-waveform');
    const cardMimicResult = document.getElementById('card-mimic-result');
    const textMimicVerdict = document.getElementById('text-mimic-verdict');
    const textMimicScore = document.getElementById('text-mimic-score');
    const barMimicProgress = document.getElementById('bar-mimic-progress');
    const textMimicPoints = document.getElementById('text-mimic-points');

    // Modals
    const modalLeaderboard = document.getElementById('modal-leaderboard');
    const btnCloseLeaderboard = document.getElementById('btn-close-leaderboard');
    const leaderboardListEl = document.getElementById('leaderboard-list');

    const modalWrongSpeaker = document.getElementById('modal-wrong-speaker');
    const btnCloseWrongSpeaker = document.getElementById('btn-close-wrong-speaker');
    const speakerSelectListEl = document.getElementById('speaker-select-list');

    const modalReset = document.getElementById('modal-reset');
    const btnCloseReset = document.getElementById('btn-close-reset');
    const btnConfirmResetScores = document.getElementById('btn-confirm-reset-scores');
    const btnConfirmResetAll = document.getElementById('btn-confirm-reset-all');

    // Toast Container
    const toastContainer = document.getElementById('toast-container');

    // -------------------------------------------------------------
    // Notification Helper
    // -------------------------------------------------------------
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className =
            'toast-enter bg-white border border-slate-200 text-slate-800 shadow-md rounded-xl px-4 py-3 text-sm flex items-center gap-2 max-w-sm pointer-events-auto';

        const dot = document.createElement('span');
        dot.className =
            type === 'success'
                ? 'w-2 h-2 rounded-full bg-emerald-500'
                : type === 'error'
                ? 'w-2 h-2 rounded-full bg-rose-500'
                : 'w-2 h-2 rounded-full bg-indigo-500';

        const msgSpan = document.createElement('span');
        msgSpan.innerText = message;

        toast.appendChild(dot);
        toast.appendChild(msgSpan);
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.25s ease';
            setTimeout(() => toast.remove(), 250);
        }, 3200);
    }

    // -------------------------------------------------------------
    // Data Loading & Roster Refresh
    // -------------------------------------------------------------
    async function refreshRoster() {
        try {
            const data = await api.listPlayers();
            state.players = data.players || [];
            renderRoster();
            updateTargetDropdown();
            updateHeaderStats();
        } catch (err) {
            console.error('Failed to load roster:', err);
        }
    }

    function updateHeaderStats() {
        const count = state.players.length;
        enrolledCountEl.innerText = count === 1 ? '1 Player' : `${count} Players`;
    }

    function renderRoster() {
        rosterListEl.innerHTML = '';
        if (state.players.length === 0) {
            rosterEmptyEl.classList.remove('hidden');
            rosterListEl.classList.add('hidden');
            return;
        }

        rosterEmptyEl.classList.add('hidden');
        rosterListEl.classList.remove('hidden');

        state.players.forEach((p) => {
            const chip = document.createElement('div');
            chip.className =
                'inline-flex items-center gap-2 bg-slate-100/80 hover:bg-slate-200/70 border border-slate-200/60 rounded-full px-3 py-1 text-xs text-slate-700 transition-colors';

            const dot = document.createElement('span');
            dot.className = 'w-1.5 h-1.5 rounded-full bg-indigo-500';

            const name = document.createElement('span');
            name.className = 'font-medium';
            name.innerText = p.name;

            const score = document.createElement('span');
            score.className = 'text-slate-400 font-normal';
            score.innerText = `${p.score} pts`;

            chip.appendChild(dot);
            chip.appendChild(name);
            chip.appendChild(score);
            rosterListEl.appendChild(chip);
        });
    }

    function updateTargetDropdown() {
        selectImpostorTarget.innerHTML = '<option value="">Select target player...</option>';
        state.players.forEach((p) => {
            const opt = document.createElement('option');
            opt.value = p.player_id;
            opt.innerText = p.name;
            selectImpostorTarget.appendChild(opt);
        });
    }

    // -------------------------------------------------------------
    // Mode Switching
    // -------------------------------------------------------------
    function switchMode(mode) {
        state.activeMode = mode;
        if (mode === 'guess_who') {
            tabGuessWho.className =
                'pill-tab px-5 py-2 rounded-full text-xs font-medium bg-white text-indigo-700 shadow-sm border border-slate-200/80';
            tabImpostor.className =
                'pill-tab px-5 py-2 rounded-full text-xs font-medium text-slate-600 hover:text-slate-900';

            sectionGuessWho.classList.remove('hidden');
            sectionImpostor.classList.add('hidden');
        } else {
            tabImpostor.className =
                'pill-tab px-5 py-2 rounded-full text-xs font-medium bg-white text-indigo-700 shadow-sm border border-slate-200/80';
            tabGuessWho.className =
                'pill-tab px-5 py-2 rounded-full text-xs font-medium text-slate-600 hover:text-slate-900';

            sectionImpostor.classList.remove('hidden');
            sectionGuessWho.classList.add('hidden');
        }
    }

    tabGuessWho.addEventListener('click', () => switchMode('guess_who'));
    tabImpostor.addEventListener('click', () => switchMode('impostor'));

    // -------------------------------------------------------------
    // Recording Flow 1: Player Registration
    // -------------------------------------------------------------
    let regCountdownTimer = null;

    btnRecordReg.addEventListener('click', async () => {
        const name = inputPlayerName.value.trim();
        if (!name) {
            showToast('Please enter a player name first.', 'error');
            inputPlayerName.focus();
            return;
        }

        if (state.isRecording) {
            // Stop recording
            await finishRegistration(name);
            return;
        }

        // Start 10-second registration recording
        try {
            await recorder.start();
            state.isRecording = true;
            state.recordingType = 'reg';

            btnRecordReg.classList.add('animate-record-pulse', 'bg-indigo-600', 'text-white');
            btnRecordReg.classList.remove('bg-indigo-50', 'text-indigo-600');
            textRecordReg.innerText = 'Recording (10s)...';

            let secondsLeft = 10;
            regCountdownTimer = setInterval(async () => {
                secondsLeft--;
                if (secondsLeft > 0) {
                    textRecordReg.innerText = `Recording (${secondsLeft}s)...`;
                } else {
                    clearInterval(regCountdownTimer);
                    await finishRegistration(name);
                }
            }, 1000);
        } catch (err) {
            showToast('Microphone access denied or unavailable.', 'error');
            console.error(err);
        }
    });

    async function finishRegistration(name) {
        if (regCountdownTimer) {
            clearInterval(regCountdownTimer);
            regCountdownTimer = null;
        }

        textRecordReg.innerText = 'Processing...';
        btnRecordReg.classList.remove('animate-record-pulse', 'bg-indigo-600', 'text-white');
        btnRecordReg.classList.add('bg-indigo-50', 'text-indigo-600');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;

            if (!blob) return;

            const res = await api.registerPlayer(name, blob);
            showToast(`Player '${res.name}' registered!`, 'success');
            inputPlayerName.value = '';
            await refreshRoster();
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            textRecordReg.innerText = 'Record Sample';
        }
    }

    // -------------------------------------------------------------
    // Recording Flow 2: Mode A (Guess Who)
    // -------------------------------------------------------------
    btnRecordGuess.addEventListener('click', async () => {
        if (state.players.length === 0) {
            showToast('Enroll at least one player in the lobby first.', 'error');
            return;
        }

        if (state.isRecording) {
            await finishGuessWho();
            return;
        }

        try {
            await recorder.start(canvasGuess);
            state.isRecording = true;
            state.recordingType = 'guess';

            cardGuessResult.classList.add('hidden');
            btnRecordGuess.classList.add('animate-record-pulse', 'bg-indigo-600', 'text-white');
            btnRecordGuess.classList.remove('bg-indigo-50', 'text-indigo-600');
            textRecordGuess.innerText = 'Listening... (Tap to stop)';
        } catch (err) {
            showToast('Microphone access denied.', 'error');
            console.error(err);
        }
    });

    async function finishGuessWho() {
        textRecordGuess.innerText = 'Analyzing acoustics...';
        btnRecordGuess.classList.remove('animate-record-pulse', 'bg-indigo-600', 'text-white');
        btnRecordGuess.classList.add('bg-indigo-50', 'text-indigo-600');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;

            if (!blob) return;

            const res = await api.guessWho(blob, 0.65);
            state.currentRound = res;

            // Render result card
            textPredictedName.innerText = res.predicted_name || 'Unknown Speaker';
            textMatchConfidence.innerText = `${res.confidence_percent}% Acoustic Match`;
            textMatchMargin.innerText = res.is_certain ? 'High Confidence' : 'Low Margin';

            cardGuessResult.classList.remove('hidden');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            textRecordGuess.innerText = 'Tap to Speak';
        }
    }

    // Correct Guess
    btnGuessCorrect.addEventListener('click', async () => {
        if (!state.currentRound || !state.currentRound.predicted_player_id) return;
        try {
            const res = await api.confirmSpeaker(
                state.currentRound.predicted_player_id,
                state.currentRound.round_id,
                50
            );
            showToast(`Correct! +50 pts awarded to ${res.name}`, 'success');
            cardGuessResult.classList.add('hidden');
            state.currentRound = null;
            await refreshRoster();
        } catch (err) {
            showToast(err.message, 'error');
        }
    });

    // Wrong Guess -> Open Modal to select actual speaker
    btnGuessWrong.addEventListener('click', () => {
        if (!state.currentRound) return;
        renderSpeakerSelectList();
        modalWrongSpeaker.classList.remove('hidden');
    });

    function renderSpeakerSelectList() {
        speakerSelectListEl.innerHTML = '';
        state.players.forEach((p) => {
            const btn = document.createElement('button');
            btn.className =
                'w-full text-left px-4 py-3 rounded-xl border border-slate-200 hover:border-indigo-400 hover:bg-indigo-50/50 transition-all flex items-center justify-between text-sm';

            const name = document.createElement('span');
            name.className = 'font-medium text-slate-800';
            name.innerText = p.name;

            const pts = document.createElement('span');
            pts.className = 'text-xs text-slate-400';
            pts.innerText = `${p.score} pts`;

            btn.appendChild(name);
            btn.appendChild(pts);

            btn.addEventListener('click', async () => {
                try {
                    const res = await api.confirmSpeaker(
                        p.player_id,
                        state.currentRound ? state.currentRound.round_id : null,
                        50
                    );
                    showToast(`Round claimed! +50 pts to ${res.name}`, 'success');
                    modalWrongSpeaker.classList.add('hidden');
                    cardGuessResult.classList.add('hidden');
                    state.currentRound = null;
                    await refreshRoster();
                } catch (err) {
                    showToast(err.message, 'error');
                }
            });

            speakerSelectListEl.appendChild(btn);
        });
    }

    btnCloseWrongSpeaker.addEventListener('click', () => {
        modalWrongSpeaker.classList.add('hidden');
    });

    // -------------------------------------------------------------
    // Recording Flow 3: Mode B (Impostor Challenge)
    // -------------------------------------------------------------
    btnRecordMimic.addEventListener('click', async () => {
        const targetId = selectImpostorTarget.value;
        if (!targetId) {
            showToast('Please select a target player to imitate.', 'error');
            selectImpostorTarget.focus();
            return;
        }

        if (state.isRecording) {
            await finishMimicChallenge(targetId);
            return;
        }

        try {
            await recorder.start(canvasMimic);
            state.isRecording = true;
            state.recordingType = 'mimic';

            cardMimicResult.classList.add('hidden');
            btnRecordMimic.classList.add('animate-record-pulse', 'bg-indigo-600', 'text-white');
            btnRecordMimic.classList.remove('bg-indigo-50', 'text-indigo-600');
            textRecordMimic.innerText = 'Mimicking... (Tap to stop)';
        } catch (err) {
            showToast('Microphone access denied.', 'error');
            console.error(err);
        }
    });

    async function finishMimicChallenge(targetId) {
        textRecordMimic.innerText = 'Evaluating mimicry...';
        btnRecordMimic.classList.remove('animate-record-pulse', 'bg-indigo-600', 'text-white');
        btnRecordMimic.classList.add('bg-indigo-50', 'text-indigo-600');

        try {
            const blob = await recorder.stop();
            state.isRecording = false;
            state.recordingType = null;

            if (!blob) return;

            const res = await api.mimicChallenge(targetId, blob, 0.60, 0.82);

            // Render result card
            textMimicVerdict.innerText = res.message;
            textMimicScore.innerText = `${res.similarity_percent}%`;
            barMimicProgress.style.width = `${Math.min(100, Math.max(0, res.similarity_percent))}%`;
            textMimicPoints.innerText = `+${res.points} pts`;

            if (res.security_breached) {
                textMimicVerdict.className = 'text-lg font-semibold text-indigo-700';
                barMimicProgress.className = 'h-full bg-indigo-600 rounded-full transition-all duration-500';
            } else if (res.status === 'CLOSE_MIMIC') {
                textMimicVerdict.className = 'text-lg font-semibold text-slate-800';
                barMimicProgress.className = 'h-full bg-slate-600 rounded-full transition-all duration-500';
            } else {
                textMimicVerdict.className = 'text-lg font-semibold text-slate-500';
                barMimicProgress.className = 'h-full bg-slate-400 rounded-full transition-all duration-500';
            }

            cardMimicResult.classList.remove('hidden');
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            textRecordMimic.innerText = 'Attempt Mimicry';
        }
    }

    // -------------------------------------------------------------
    // Leaderboard Modal
    // -------------------------------------------------------------
    btnOpenLeaderboard.addEventListener('click', async () => {
        try {
            const data = await api.getScoreboard();
            renderLeaderboard(data.players || []);
            modalLeaderboard.classList.remove('hidden');
        } catch (err) {
            showToast('Failed to load leaderboard.', 'error');
        }
    });

    btnCloseLeaderboard.addEventListener('click', () => {
        modalLeaderboard.classList.add('hidden');
    });

    function renderLeaderboard(players) {
        leaderboardListEl.innerHTML = '';
        if (players.length === 0) {
            leaderboardListEl.innerHTML =
                '<p class="text-xs text-slate-400 text-center py-6">No scores recorded yet.</p>';
            return;
        }

        players.forEach((p, idx) => {
            const row = document.createElement('div');
            row.className =
                'flex items-center justify-between py-2.5 px-3 rounded-xl hover:bg-slate-50 text-sm transition-colors';

            const left = document.createElement('div');
            left.className = 'flex items-center gap-3';

            const rank = document.createElement('span');
            rank.className = 'text-xs font-medium text-slate-400 w-4';
            rank.innerText = `#${idx + 1}`;

            const name = document.createElement('span');
            name.className = 'font-medium text-slate-800';
            name.innerText = p.name;

            left.appendChild(rank);
            left.appendChild(name);

            const score = document.createElement('span');
            score.className = 'font-semibold text-indigo-700';
            score.innerText = `${p.score} pts`;

            row.appendChild(left);
            row.appendChild(score);
            leaderboardListEl.appendChild(row);
        });
    }

    // -------------------------------------------------------------
    // Reset Modal
    // -------------------------------------------------------------
    btnOpenReset.addEventListener('click', () => {
        modalReset.classList.remove('hidden');
    });

    btnCloseReset.addEventListener('click', () => {
        modalReset.classList.add('hidden');
    });

    btnConfirmResetScores.addEventListener('click', async () => {
        try {
            await api.resetGame(true);
            showToast('Scores reset to zero.', 'info');
            modalReset.classList.add('hidden');
            await refreshRoster();
        } catch (err) {
            showToast('Reset failed.', 'error');
        }
    });

    btnConfirmResetAll.addEventListener('click', async () => {
        try {
            await api.resetGame(false);
            showToast('All players and games cleared.', 'info');
            modalReset.classList.add('hidden');
            cardGuessResult.classList.add('hidden');
            cardMimicResult.classList.add('hidden');
            await refreshRoster();
        } catch (err) {
            showToast('Reset failed.', 'error');
        }
    });

    // Initial Load
    refreshRoster();
});
