from PIL import Image
from bs4 import BeautifulSoup
from PIL import Image
import cairosvg
from io import BytesIO

CIRCLE_SVG = "<svg><circle cx='50%' cy='50%' r='50%' /></svg>"
VOID_SVF = "<svg></svg>"


def clean_svg(svg_text, output_width=None, output_height=None):
    soup = BeautifulSoup(svg_text, 'xml') # Read as soup to parse as xml
    svg_bs4 = soup.prettify() # Prettify to get a string

    # Store the original signal handler
    import signal
    original_handler = signal.getsignal(signal.SIGALRM)
    
    try:
        # Set a timeout to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("SVG processing timed out")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        # Try direct conversion without BeautifulSoup
        svg_cairo = cairosvg.svg2svg(svg_bs4, output_width=output_width, output_height=output_height).decode()
        
    except TimeoutError:
        print("SVG conversion timed out, using fallback method")
        svg_cairo = """<svg></svg>"""
    finally:
        # Always cancel the alarm and restore original handler, regardless of success or failure
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        
    svg_clean = "\n".join([line for line in svg_cairo.split("\n") if not line.strip().startswith("<?xml")]) # Remove xml header
    return svg_clean

def rasterize_svg(svg_string, resolution=224, dpi = 128, scale=2):
    try:
        svg_raster_bytes = cairosvg.svg2png(
            bytestring=svg_string,
            background_color='white',
            output_width=resolution, 
            output_height=resolution,
            dpi=dpi,
            scale=scale) 
        svg_raster = Image.open(BytesIO(svg_raster_bytes))
    except: 
        try:
            svg = clean_svg(svg_string)
            svg_raster_bytes = cairosvg.svg2png(
                bytestring=svg,
                background_color='white',
                output_width=resolution, 
                output_height=resolution,
                dpi=dpi,
                scale=scale) 
            svg_raster = Image.open(BytesIO(svg_raster_bytes))
        except:
            svg_raster = Image.new('RGB', (resolution, resolution), color = 'white')
    return svg_raster
    
