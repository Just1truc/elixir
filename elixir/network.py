import io
import torch
import base64
import torch as t

class NetworkTensor:
    
    def __init__(
        self,
        data : str | t.Tensor
    ):
        """
        Args:
            data (str | t.Tensor): If it's a tensor, then we will serialize it
            # If it's a serialized version, we deserialize it
        """
        self.data = data

    def deserialize(self):
        
        def apply(data : str):
            b = base64.b64decode(data.encode("ascii"))
            return torch.load(io.BytesIO(b), map_location="cpu")

        return self.data if isinstance(self.data, t.Tensor) else apply(self.data)
        
    def serialize(self):
        
        def apply(data : t.Tensor):
            buf = io.BytesIO()
            t.save(data, buf)
            
            return base64.b64encode(buf.getvalue()).decode("ascii")
        
        return self.data if isinstance(self.data, str) else apply(self.data)