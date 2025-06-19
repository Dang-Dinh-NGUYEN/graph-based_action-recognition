class Joint:
    def __init__(self, jointinfo, tracking_state):
        self.x, self.y, self.z = jointinfo[0:3]
        self.depthX, self.depthY = jointinfo[3:5]
        self.colorX, self.colorY = jointinfo[5:7]
        self.orientationW = jointinfo[7]
        self.orientationX = jointinfo[8]
        self.orientationY = jointinfo[9]
        self.orientationZ = jointinfo[10]
        self.trackingState = tracking_state