# /// script
# dependencies = [
#   "pypdfium2",
#   "matplotlib",
#   "spacy-layout",
# ]
# ///

import pypdfium2 as pdfium
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import spacy
from scripts.notebooks.spacy_layout_test import spaCyLayout

DOCUMENT_PATH = "/Users/kun/Downloads/1.pdf"

# Load and convert the PDF page to an image
pdf = pdfium.PdfDocument(DOCUMENT_PATH)
page_image = pdf[2].render(scale=1)  # get page 3 (index 2)
numpy_array = page_image.to_numpy()
# Process document with spaCy
nlp = spacy.blank("en")
layout = spaCyLayout(nlp)
doc = layout(DOCUMENT_PATH)

# Get page 3 layout and sections
page = doc._.pages[2]
page_layout = doc._.layout.pages[2]
# Create figure and axis with page dimensions
fig, ax = plt.subplots(figsize=(12, 16))
# Display the PDF image
ax.imshow(numpy_array)
# Add rectangles for each section's bounding box
for section in page[1]:
    # Create rectangle patch
    rect = Rectangle(
        (section._.layout.x, section._.layout.y),
        section._.layout.width,
        section._.layout.height,
        fill=False,
        color="blue",
        linewidth=1,
        alpha=0.5
    )
    ax.add_patch(rect)
    # Add text label at top of box
    ax.text(
        section._.layout.x,
        section._.layout.y,
        section.label_,
        fontsize=8,
        color="red",
        verticalalignment="bottom"
    )

ax.axis("off")  # hide axes
plt.show()