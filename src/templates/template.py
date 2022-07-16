import cv2

class Template:
    def __init__(self) -> None:
        self.image = None
        self.mask = None
        self.connections = []
        self.name = ""
    
    def __init__(self, image, mask, name, connections) -> None:
        self.image = image
        self.mask = mask
        self.connections = connections
        self.name = name
    
    def rotate_cw(self):
        self.connections = [[[i[1],self.image.shape[0] -  i[0]] for i in j] for j in self.connections]
        self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
        self.mask = cv2.rotate(self.mask, cv2.ROTATE_90_CLOCKWISE)
    
    # only if needed 
    def rotate_ccw(self):
        pass
    def scale(self, factor):
        new_img = cv2.resize(self.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        new_mask = cv2.resize(self.mask, None, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        new_connections =  [[[int(i[0]*factor), int(i[1]*factor)] for i in j] for j in self.connections]
        return Template(new_img, new_mask, self.name, new_connections)
    def rotate(self, angle):
        pass
    
    def display(self, plt):
        plt.imshow(self.image)
        plt.show()
        plt.imshow(self.mask)
        plt.show()
