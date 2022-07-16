from PyPDF2 import PdfReader

reader = PdfReader("zoet2.pdf")
page  = reader.pages[157]
# extract all images from page

print(page.getObject('/XObject')) 