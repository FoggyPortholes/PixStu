import os
from app.pixel_char_studio import demo
port = int(os.getenv("PCS_PORT","7860"))
demo.launch(share=False, inbrowser=True, server_name="127.0.0.1", server_port=port, show_error=True)
