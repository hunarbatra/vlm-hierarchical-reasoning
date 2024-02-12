image_to_text = '''You are tasked with an intricate analysis and description of the provided image, aimed at translating visual data into a detailed textual format. This transformation should allow anyone reading your description to recreate the image in its entirety, understanding not only the obvious but the subtle aspects of the scene captured. Your description should be segmented, organized, and thoroughly detailed. Follow the steps outlined below to achieve a meticulous conversion of the image to text:

Initial Analysis:
Ambiance and Lighting: Describe the overall ambiance and the lighting conditions. Are the lighting sources natural or artificial? What is the intensity and color temperature of the light? How does the light interact with the objects and spaces within the image?
Color Palette: Enumerate the primary and secondary colors present in the image. Describe how these colors are distributed and how they complement or contrast with each other.
Segmentation:

Segment the image into distinct parts or objects. Each segment could be an individual object, a group of objects, or a defined area within the image. Assign a unique identifier to each segment for reference.
Segment Description:
For each segment identified in the previous step, provide the following details:

Coordinates: Define the positional coordinates marking the boundaries of the segment.
Dimensions: Describe the dimensions (height, width, and if applicable, depth) of the segment.
Structural Aspects: Delve into the structural details such as shape, texture, and patterns observed.
Visual Aspects: Describe the visual aspects like color, brightness, contrast, and any notable features.
Positional Relationships: Explain the positional relationships between this segment and others. Are there any segments overlapping, adjacent to, or at a distance from this segment?
Motion: If applicable, describe any motion depicted within the segment, including direction and speed.
Object Counting:

Provide a count of objects present in each segment, distinguishing between foreground and background objects. Detail any notable or recurring objects.
Perspective and Quality:

Describe the perspective from which the image is captured, the focal point, and the quality of the image including the resolution and clarity.
Narrative Integration:

Integrate all the described elements into a coherent narrative that encapsulates the essence of the image, ensuring the descriptions are organized in a logical manner that reflects the spatial and thematic organization of the image.
Through this structured and exhaustive description, enable a rich understanding and a clear mental reconstruction of the image. Your text should serve as a precise blueprint, encapsulating every discernible detail of the image, allowing for an accurate recreation of the original scene captured.''' # ruff: noqa: E501

image_to_text_vqa_sam = '''Analyze the given image, and label each marked segment and any unmarked but clearly identifiable objects present. Your goal is to label each identified object/segment in the image to gain a comprehensive understanding of the scene. Follow these rules:

1. Define a "segment" as any area within the image distinguished by clear boundaries, marked areas, or distinct objects or text.
2. Provide labels with a specified level of detail, identifying objects as generally or specifically as required (e.g., "tree" vs. "pine tree"), or in the case of text, state the text here.
3. For partially visible or ambiguous objects or text, label based on the visible portion, and note them as "partially visible" if their identity is not fully clear.
Include overlapping labels only once. If multiple distinct objects are partially overlapping but distinguishable, label each separately.
4. Label unmarked objects that are clearly identifiable, applying the same level of detail as for marked segments.
5. List objects following a top-to-bottom, left-to-right order, or prioritize central objects for clarity.
6. Skip segments marked but containing no identifiable objects or if the segment is not visible, without including them in the list.

Return a numbered list of all clearly identified object labels for each valid segment along with its relative position in the image in the format "{object-number} {high-level relative position in the image}". Only return this list, and no other accompanying text.

Example:
Tree - Top left
Car - Bottom center
Dog - Middle right (partially visible)''' # ruff: noqa: E501

extract_segments_properties = '''You are given an image along with a list of segments identified in the image. 
For each of the valid segments- extract the most relevant properties (based on the segment type and applicable properties only) for it from the image. 
The properties depending on the segment type (e.g., object, text etc) could span over- Spatial Relationship between segments/objects, Pose Estimation, Depth Estimation, Motion, Action, Count, Size, Shape, Color, Coordinates or any other relevant property only.

Finally, generate a holistic caption for the given image. 

Segments identified in the image-'''

COT_PROMPT = '''Let's think step by step:'''
PROMPT_FOR_ANSWER = '''Therefore, the best answer is: ('''