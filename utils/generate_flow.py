import xml.etree.ElementTree as ET

file = 'D:/Paper Publish/MARL_TSC/MARL_TSC_code/scenarios/h_corridor/sample-400.rou.xml'

# Split and modify the probability values for <flow> elements
mpr = 0.8


def indent(elem, level=0):
    """In-place pretty print XML element with indentation."""
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# Parse the original XML file
tree = ET.parse(file)
root = tree.getroot()

# Create a new root for the modified XML
new_root = ET.Element('routes')

# Copy <route> elements to the new XML
for route_elem in root.findall('.//route'):
    new_root.append(route_elem)

# Create a separate root for the modified <flow> elements
flow_root = ET.Element('flows')

for flow_elem in root.findall('.//flow'):
    original_probability = float(flow_elem.get('probability'))

    # Iterate over two cases: (1) type "car" and (2) type "cv"
    for i, type_value in enumerate(['cv', 'car']):
        new_flow_elem = ET.Element('flow')
        new_flow_elem.set('id', f"{flow_elem.get('id')}_{i}")
        new_flow_elem.set('begin', flow_elem.get('begin'))
        new_flow_elem.set('end', flow_elem.get('end'))
        new_flow_elem.set('route', flow_elem.get('route'))
        new_flow_elem.set('departLane', flow_elem.get('departLane'))
        new_flow_elem.set('departSpeed', flow_elem.get('departSpeed'))
        new_flow_elem.set('type', type_value)  # New "type" attribute

        # Calculate split probability based on the index and round to 5 decimal places
        split_probability = round(original_probability * (mpr if i == 0 else (1 - mpr)),
                                  5)
        new_flow_elem.set('probability', str(split_probability))

        # Append the new <flow> element to the flow_root
        flow_root.append(new_flow_elem)

# Append the flow_root to the new_root
new_root.append(flow_root)

# Create a new tree with the modified XML data
new_tree = ET.ElementTree(new_root)

# In-place pretty print the modified XML
indent(new_tree.getroot())

# Write the new XML data to a new file
with open(f'D:/Paper Publish/MARL_TSC/MARL_TSC_code/scenarios/h_corridor/cv{mpr}_offpeak.rou.xml', 'w', encoding='utf-8') as f:
    f.write(ET.tostring(new_tree.getroot(), encoding='utf-8').decode('utf-8'))
