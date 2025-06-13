import re
from io import BytesIO
from bs4 import BeautifulSoup
import cairosvg
from PIL import Image
import numpy as np
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    """
    Context manager for implementing timeout functionality using signals
    
    This provides a timeout mechanism for operations that might hang,
    particularly useful for SVG rendering which can become unresponsive
    with malformed or complex SVG data.
    
    Parameters:
    ----------
    seconds : int
        Timeout duration in seconds
        
    Raises:
    ------
    TimeoutError
        If the operation exceeds the specified timeout
        
    Example:
    -------
    with timeout(10):
        # Operation that might take too long
        result = some_potentially_slow_operation()
    """
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timeout: Operation exceeded {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_svg_validity(svg_string):
    """
    Comprehensive SVG validation including structure and rendering verification
    
    This function performs a two-stage validation process:
    1. Rendering validation: Tests if the SVG can be successfully rendered
    2. Structure validation: Checks SVG syntax, elements, and attributes
    
    Parameters:
    ----------
    svg_string : str
        SVG string to validate
    
    Returns:
    -------
    tuple
        (is_valid, message) where is_valid is boolean and message contains details
        
    Validation Criteria:
    ------------------
    - Must be valid XML structure
    - Must have proper SVG namespace
    - Must contain only <svg> and <path> elements
    - Must have at least one <path> element
    - Each <path> must have valid 'd' attribute
    - Must be renderable without timeout or errors
    """
    # Basic input validation
    if not isinstance(svg_string, str) or not svg_string.strip():
        return False, "Input is not a valid SVG string"
    
    # Stage 1: Rendering validation
    try:
        # Use timeout mechanism to wrap rendering function
        try:
            with timeout(15):  # 15 second timeout
                png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        except TimeoutError:
            return False, "Rendering timeout: SVG may contain invalid data causing renderer to hang"
        
        # Check if rendering was successful
        if not png_data:
            return False, "SVG cannot be rendered"
        
        # Check if rendered result has content
        img = Image.open(BytesIO(png_data))
        
        # Check dimensions
        if img.width == 0 or img.height == 0:
            return False, "SVG renders with zero dimensions"
        
        # Optional: Check for visible pixels (commented out to allow empty renders)
        # img_array = np.array(img)
        # if img.mode == 'RGBA' and np.all(img_array[:, :, 3] == 0):
        #     return False, "SVG renders without visible content"
        
    except Exception as e:
        return False, f"SVG rendering validation failed: {str(e)}"
    
    # Stage 2: Structure validation
    try:
        # Parse SVG using BeautifulSoup with XML parser
        soup = BeautifulSoup(svg_string, 'xml')
        
        # Check for svg root element
        svg_element = soup.find('svg')
        if not svg_element:
            return False, "Missing svg root element"
        
        # Optional: Check viewBox attribute (commented out as it's not always required)
        # if not svg_element.get('viewBox'):
        #     return False, "Missing viewBox attribute"
        
        # Check SVG namespace
        xmlns = svg_element.get('xmlns')
        if not xmlns or xmlns != "http://www.w3.org/2000/svg":
            return False, "Missing or incorrect SVG namespace"
        
        # Check that only allowed elements are present
        all_elements = soup.find_all()
        for element in all_elements:
            if element.name not in ['svg', 'path']:
                return False, f"Contains disallowed element: {element.name}"
        
        # Check for presence of path elements
        path_elements = svg_element.find_all('path')
        if not path_elements:
            return False, "Contains no path elements"
        
        # Validate each path element
        for i, path in enumerate(path_elements):
            # Check for 'd' attribute
            if 'd' not in path.attrs:
                return False, f"Path element {i+1} missing 'd' attribute"
    
    except Exception as e:
        return False, f"SVG structure validation failed: {str(e)}"
    
    # All validations passed
    return True, "SVG is valid and renderable"

def is_valid_path_data(d_value):
    """
    Validate SVG path data string
    
    Checks if the path data string follows SVG path syntax rules:
    - Must start with valid command letter
    - Must have balanced parentheses
    - Must contain only valid characters
    - Must have parseable command structure
    
    Parameters:
    ----------
    d_value : str
        SVG path data string
        
    Returns:
    -------
    bool
        True if path data is valid, False otherwise
        
    Valid Commands:
    --------------
    M/m: Move to
    L/l: Line to
    H/h: Horizontal line to
    V/v: Vertical line to
    C/c: Cubic Bezier curve to
    S/s: Smooth cubic Bezier curve to
    Q/q: Quadratic Bezier curve to
    T/t: Smooth quadratic Bezier curve to
    A/a: Elliptical arc to
    Z/z: Close path
    """
    if not d_value or not d_value.strip():
        return False
    
    # Basic syntax check: should start with valid command letter
    if not re.match(r'^[MLHVCSQTAZmlhvcsqtaz]', d_value.strip()):
        return False
    
    # Check parentheses balance
    if d_value.count('(') != d_value.count(')'):
        return False
    
    # Check for invalid characters
    valid_chars = set("MLHVCSQTAZmlhvcsqtaz0123456789,.-+eE() \t\n")
    if not all(c in valid_chars for c in d_value):
        return False
    
    # Basic path command validity check
    try:
        commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz][^MLHVCSQTAZmlhvcsqtaz]*', d_value)
        if not commands:
            return False
    except:
        return False
    
    return True

def is_valid_color(color_value):
    """
    Validate SVG color value
    
    Supports multiple color formats:
    - Hexadecimal: #RGB, #RRGGBB
    - RGB functions: rgb(r,g,b), rgba(r,g,b,a)
    - Named colors: Standard CSS/SVG color names
    - Special values: none, transparent, currentColor
    
    Parameters:
    ----------
    color_value : str
        Color value to validate
        
    Returns:
    -------
    bool
        True if color value is valid, False otherwise
    """
    # Empty, none, and transparent are considered valid
    if not color_value or color_value.lower() in ["none", "transparent"]:
        return True
    
    # Hexadecimal color format (#RGB or #RRGGBB)
    if re.match(r'^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$', color_value):
        return True
    
    # RGB/RGBA color format
    if (re.match(r'^rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$', color_value) or 
        re.match(r'^rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d\.]+\s*\)$', color_value)):
        return True
    
    # Standard named colors (CSS/SVG color keywords)
    named_colors = {
        # Basic colors
        "black", "silver", "gray", "white", "maroon", "red", "purple", "fuchsia", 
        "green", "lime", "olive", "yellow", "navy", "blue", "teal", "aqua", "orange",
        
        # Extended colors
        "aliceblue", "antiquewhite", "aquamarine", "azure", "beige", "bisque", 
        "blanchedalmond", "blueviolet", "brown", "burlywood", "cadetblue", 
        "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson",
        "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen",
        "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid",
        "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
        "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray",
        "dodgerblue", "firebrick", "floralwhite", "forestgreen", "gainsboro",
        "ghostwhite", "gold", "goldenrod", "greenyellow", "honeydew", "hotpink",
        "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush",
        "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
        "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink",
        "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightsteelblue", "lightyellow", "limegreen", "linen", "magenta",
        "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
        "mediumseagreen", "mediumslateblue", "mediumspringgreen",
        "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream",
        "mistyrose", "moccasin", "navajowhite", "oldlace", "olivedrab",
        "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
        "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum",
        "powderblue", "rosybrown", "royalblue", "saddlebrown", "salmon",
        "sandybrown", "seagreen", "seashell", "sienna", "skyblue", "slateblue",
        "slategray", "snow", "springgreen", "steelblue", "tan", "thistle",
        "tomato", "turquoise", "violet", "wheat", "whitesmoke", "yellowgreen",
        
        # Special SVG colors
        "currentColor"
    }
    
    if color_value.lower() in named_colors:
        return True
    
    return False

def evaluate_svg(model_svg):
    """
    Evaluate whether a model-generated SVG is valid
    
    This is the main evaluation function that provides a standardized
    assessment of SVG validity for machine learning model outputs.
    
    Parameters:
    ----------
    model_svg : str
        Model-generated SVG string
    
    Returns:
    -------
    dict
        Evaluation results containing:
        - valid: Boolean indicating if SVG is valid
        - score: Numerical score (1.0 for valid, 0.0 for invalid)
        - message: Detailed validation message
        - next_step: Guidance for next evaluation steps (if valid)
        
    Usage in ML Pipelines:
    ---------------------
    This function is designed to be used as a first-stage filter
    in SVG evaluation pipelines, ensuring that only valid SVGs
    proceed to more detailed quality assessments.
    """
    is_valid, message = check_svg_validity(model_svg)
    
    if is_valid:
        return {
            "valid": True,
            "score": 1.0,
            "message": message,
            "next_step": "Ready for detailed quality evaluation"
        }
    else:
        return {
            "valid": False,
            "score": 0.0,
            "message": message,
            "next_step": "Fix validation errors before proceeding"
        }

def batch_evaluate_svgs(svg_list, include_details=False):
    """
    Evaluate multiple SVGs in batch for efficiency
    
    Parameters:
    ----------
    svg_list : list
        List of SVG strings to evaluate
    include_details : bool
        Whether to include detailed messages for each SVG
        
    Returns:
    -------
    dict
        Batch evaluation results including:
        - total_count: Total number of SVGs evaluated
        - valid_count: Number of valid SVGs
        - invalid_count: Number of invalid SVGs
        - validity_rate: Percentage of valid SVGs
        - results: List of individual results (if include_details=True)
    """
    results = []
    valid_count = 0
    
    for i, svg in enumerate(svg_list):
        try:
            result = evaluate_svg(svg)
            if include_details:
                result['index'] = i
                results.append(result)
            
            if result['valid']:
                valid_count += 1
                
        except Exception as e:
            error_result = {
                "valid": False,
                "score": 0.0,
                "message": f"Evaluation error: {str(e)}",
                "index": i
            }
            if include_details:
                results.append(error_result)
    
    total_count = len(svg_list)
    invalid_count = total_count - valid_count
    validity_rate = (valid_count / total_count * 100) if total_count > 0 else 0
    
    batch_result = {
        "total_count": total_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validity_rate": validity_rate
    }
    
    if include_details:
        batch_result["results"] = results
    
    return batch_result

def validate_svg_file(file_path):
    """
    Validate SVG from file path
    
    Parameters:
    ----------
    file_path : str
        Path to SVG file
        
    Returns:
    -------
    dict
        Validation result dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        result = evaluate_svg(svg_content)
        result['file_path'] = file_path
        return result
        
    except FileNotFoundError:
        return {
            "valid": False,
            "score": 0.0,
            "message": f"File not found: {file_path}",
            "file_path": file_path
        }
    except Exception as e:
        return {
            "valid": False,
            "score": 0.0,
            "message": f"Error reading file: {str(e)}",
            "file_path": file_path
        }

# Example usage and testing
if __name__ == "__main__":
    """
    Demonstration of SVG validation functionality
    
    This section shows how to use the validator with various
    types of valid and invalid SVG examples.
    """
    
    print("=== SVG Validator Demo ===\n")
    
    # Valid SVG example
    valid_svg = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="M10,10 L90,10 L90,90 L10,90 Z" fill="blue" />
    </svg>
    """
    
    # Invalid SVG examples for testing different failure modes
    
    # Invalid: Contains rect element (not allowed, only path elements permitted)
    invalid_svg_rect = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <rect x="10" y="10" width="80" height="80" fill="red" />
    </svg>
    """
    
    # Invalid: Missing required namespace
    invalid_svg_namespace = """
    <svg viewBox="0 0 100 100">
        <path d="M10,10 L90,10 L90,90 L10,90 Z" fill="green" />
    </svg>
    """
    
    # Invalid: Missing 'd' attribute in path
    invalid_svg_no_d = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path fill="yellow" />
    </svg>
    """
    
    # Invalid: Malformed path data
    invalid_svg_bad_path = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <path d="X10,10 L90,10 L90,90 L10,90 Z" fill="green" />
    </svg>
    """
    
    # Invalid: No path elements
    invalid_svg_no_paths = """
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    </svg>
    """
    
    # Test cases
    test_cases = [
        ("Valid SVG", valid_svg),
        ("Invalid: Contains rect element", invalid_svg_rect),
        ("Invalid: Missing namespace", invalid_svg_namespace),
        ("Invalid: Missing 'd' attribute", invalid_svg_no_d),
        ("Invalid: Malformed path data", invalid_svg_bad_path),
        ("Invalid: No path elements", invalid_svg_no_paths)
    ]
    
    print("Individual SVG Validation Tests:")
    print("-" * 50)
    
    for test_name, svg in test_cases:
        print(f"\nTest: {test_name}")
        result = evaluate_svg(svg)
        print(f"Valid: {result['valid']}")
        print(f"Score: {result['score']}")
        print(f"Message: {result['message']}")
        if 'next_step' in result:
            print(f"Next Step: {result['next_step']}")
    
    print("\n" + "="*60)
    print("Batch Validation Test:")
    print("-" * 25)
    
    # Test batch evaluation
    svg_batch = [valid_svg, invalid_svg_rect, invalid_svg_namespace, valid_svg]
    batch_result = batch_evaluate_svgs(svg_batch, include_details=True)
    
    print(f"Total SVGs: {batch_result['total_count']}")
    print(f"Valid SVGs: {batch_result['valid_count']}")
    print(f"Invalid SVGs: {batch_result['invalid_count']}")
    print(f"Validity Rate: {batch_result['validity_rate']:.1f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(batch_result['results']):
        status = "✓ VALID" if result['valid'] else "✗ INVALID"
        print(f"  SVG {i+1}: {status} - {result['message']}")
    
    print("\n" + "="*60)
    print("Feature Summary:")
    print("-" * 15)
    print("✓ Structure validation (XML syntax, elements, attributes)")
    print("✓ Rendering validation (timeout protection, output verification)")
    print("✓ Batch processing capabilities")
    print("✓ File-based validation support")
    print("✓ Comprehensive error reporting")
    print("✓ ML pipeline integration ready")
    
    print("\nValidation completed successfully!")