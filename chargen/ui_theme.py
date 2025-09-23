PIXEL_THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');

:root {
    --chargen-bg: #0d0f23;
    --chargen-panel: rgba(20, 24, 52, 0.92);
    --chargen-neon: #3cffd0;
    --chargen-neon-alt: #ff33a8;
    --chargen-text: #e6f5ff;
    --chargen-muted: rgba(230, 245, 255, 0.65);
    --chargen-border: rgba(60, 255, 208, 0.38);
    font-family: 'Press Start 2P', monospace;
}

body {
    font-family: 'Press Start 2P', monospace;
    background: radial-gradient(circle at top, rgba(63, 63, 102, 0.45), transparent 55%),
                repeating-linear-gradient(0deg, rgba(60, 255, 208, 0.08) 0, rgba(60, 255, 208, 0.08) 2px, transparent 2px, transparent 8px),
                #0d0f23;
    color: var(--chargen-text);
}

.gradio-container {
    background-color: transparent !important;
}

#chargen-tabs .tabitem,
#chargen-tabs .tabitem label,
#chargen-tabs .tabitem button,
#chargen-tabs .tabitem input,
#chargen-tabs .tabitem textarea,
#chargen-tabs .tabitem select {
    font-family: 'Press Start 2P', monospace;
}

#chargen-tabs .tab-nav button {
    border: 2px solid var(--chargen-neon);
    color: var(--chargen-neon);
    background: rgba(20, 24, 52, 0.85);
}

#chargen-tabs .tab-nav button.selected {
    background: linear-gradient(135deg, rgba(60, 255, 208, 0.4), rgba(255, 51, 168, 0.45));
    color: var(--chargen-text);
}

.chargen-panel {
    background: var(--chargen-panel);
    border: 2px solid var(--chargen-border);
    box-shadow: 0 0 24px rgba(60, 255, 208, 0.28), inset 0 0 12px rgba(255, 51, 168, 0.25);
    padding: 16px;
    border-radius: 8px;
}

.chargen-group-title {
    color: var(--chargen-neon);
    text-shadow: 0 0 8px rgba(60, 255, 208, 0.65);
    margin-bottom: 12px;
}

button.chargen-primary {
    border: 2px solid var(--chargen-neon);
    background: linear-gradient(135deg, rgba(60, 255, 208, 0.65), rgba(255, 51, 168, 0.55));
    color: var(--chargen-bg);
    text-transform: uppercase;
}

button.chargen-primary:hover {
    filter: brightness(1.08);
}

#reference-gallery .thumbnail-item {
    border: 2px solid rgba(60, 255, 208, 0.4);
}

.chargen-hint {
    font-size: 11px;
    color: var(--chargen-muted);
    margin-top: 4px;
}
"""


def theme_css() -> str:
    return PIXEL_THEME_CSS
