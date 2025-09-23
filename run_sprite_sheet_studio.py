import os

if __name__ == "__main__":
    from app.sprite_sheet_studio import demo

    port = int(os.getenv("PCS_SPRITE_SHEET_PORT", "7865"))
    server_name = os.getenv("PCS_SERVER_NAME", "127.0.0.1")
    open_browser = os.getenv("PCS_OPEN_BROWSER", "0").lower() in {"1", "true", "yes", "on"}
    demo.launch(
        share=False,
        inbrowser=open_browser,
        server_name=server_name,
        server_port=port,
        show_error=True,
    )
