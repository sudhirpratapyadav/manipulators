"""
AIModel Actor - base class for ML model inference actors.

Provides framework for integrating AI models (vision models, policy networks, etc.)
Subclasses implement specific model interfaces (PyTorch, ONNX, TensorFlow, etc.)
"""

import queue
import time
from abc import abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from ..core.actor import Actor
from ..core.bus import MessageBus, Topics
from ..core.messages import InferenceRequest, InferenceResult
from ..core.config import Config


class AIModelActor(Actor):
    """
    Base class for AI model inference actors.

    Subscribes: /ai/inference/request
    Publishes: /ai/inference/result/{model_id}
    """

    def __init__(self, model_id: str, bus: MessageBus, config: Config):
        super().__init__(
            name=f"AIModelActor_{model_id}",
            bus=bus,
            config=config
        )

        self.model_id = model_id
        self._config = config

        # Topics
        self.result_topic = Topics.AI_INFERENCE_RESULT.replace("{model_id}", model_id)

        # Queues
        self._request_queue: Optional[queue.Queue] = None

        # Model state
        self.model = None
        self.device = "cpu"

    @abstractmethod
    def load_model(self) -> bool:
        """
        Load ML model from disk/checkpoint.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def preprocess(self, input_data: Dict[str, Any]) -> Any:
        """
        Preprocess input data for inference.

        Args:
            input_data: Raw input data dictionary

        Returns:
            Preprocessed data in format expected by model
        """
        pass

    @abstractmethod
    def infer(self, preprocessed_data: Any) -> Any:
        """
        Run inference on preprocessed data.

        Args:
            preprocessed_data: Output from preprocess()

        Returns:
            Raw model output
        """
        pass

    @abstractmethod
    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """
        Postprocess model output into standard format.

        Args:
            model_output: Raw output from model

        Returns:
            Dictionary with processed results
        """
        pass

    @abstractmethod
    def cleanup_model(self) -> None:
        """Cleanup model resources."""
        pass

    def setup(self) -> None:
        """Initialize AI model actor."""
        self._request_queue = self.bus.subscribe_queue(Topics.AI_INFERENCE_REQUEST, maxsize=10)

        success = self.load_model()
        if not success:
            print(f"[{self.name}] Failed to load model")

    def loop(self) -> None:
        """Process inference requests."""
        # Get all pending requests
        try:
            while True:
                req: InferenceRequest = self._request_queue.get_nowait()

                # Only process requests for this model
                if req.model_id != self.model_id:
                    continue

                # Run inference
                t_start = time.time()
                try:
                    preprocessed = self.preprocess(req.input_data)
                    model_output = self.infer(preprocessed)
                    output_data = self.postprocess(model_output)
                    latency_ms = (time.time() - t_start) * 1000.0

                    # Publish result
                    result = InferenceResult(
                        model_id=self.model_id,
                        request_id=req.request_id,
                        output_data=output_data,
                        latency_ms=latency_ms
                    )
                    self.bus.publish(self.result_topic, result)

                except Exception as e:
                    print(f"[{self.name}] Inference error: {e}")
                    # Publish error result
                    result = InferenceResult(
                        model_id=self.model_id,
                        request_id=req.request_id,
                        output_data={'error': str(e)},
                        confidence=0.0,
                        latency_ms=(time.time() - t_start) * 1000.0
                    )
                    self.bus.publish(self.result_topic, result)

        except queue.Empty:
            pass

        # Sleep to avoid busy-waiting
        time.sleep(0.001)  # 1ms

    def teardown(self) -> None:
        """Cleanup model resources."""
        self.cleanup_model()


# ============ Example Implementations ============

class MockVisionModelActor(AIModelActor):
    """Mock vision model for testing."""

    def __init__(self, bus: MessageBus, config: Config):
        super().__init__("mock_vision", bus, config)

    def load_model(self) -> bool:
        print(f"[{self.name}] Mock model loaded")
        self.model = "mock"
        return True

    def preprocess(self, input_data: Dict[str, Any]) -> Any:
        # Expect input_data to have 'image' key
        return input_data.get('image', np.zeros((224, 224, 3)))

    def infer(self, preprocessed_data: Any) -> Any:
        # Mock classification
        time.sleep(0.01)  # Simulate inference time
        return np.random.rand(10)  # 10 class scores

    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        scores = model_output
        pred_class = int(np.argmax(scores))
        confidence = float(scores[pred_class])

        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'class_scores': scores.tolist()
        }

    def cleanup_model(self) -> None:
        print(f"[{self.name}] Cleanup")


class PyTorchModelActor(AIModelActor):
    """
    PyTorch model actor.

    Example for loading and running PyTorch models.
    """

    def __init__(self, model_id: str, bus: MessageBus, config: Config,
                 model_path: str, device: str = "cpu"):
        super().__init__(model_id, bus, config)
        self.model_path = model_path
        self.device = device

    def load_model(self) -> bool:
        """Load PyTorch model from checkpoint."""
        try:
            import torch

            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            print(f"[{self.name}] PyTorch model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Load error: {e}")
            return False

    def preprocess(self, input_data: Dict[str, Any]) -> Any:
        """Preprocess input for PyTorch model."""
        import torch
        import torchvision.transforms as transforms

        # Example: preprocess image
        if 'image' in input_data:
            image = input_data['image']  # numpy array

            # Convert to tensor and normalize
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            tensor = transform(image).unsqueeze(0).to(self.device)
            return {'image_tensor': tensor}

        return input_data

    def infer(self, preprocessed_data: Any) -> Any:
        """Run PyTorch inference."""
        import torch

        with torch.no_grad():
            image_tensor = preprocessed_data['image_tensor']
            output = self.model(image_tensor)
            return output

    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess PyTorch output."""
        # Convert tensor to numpy
        output_np = model_output.cpu().numpy()

        return {
            'output': output_np.tolist(),
            'shape': list(output_np.shape)
        }

    def cleanup_model(self) -> None:
        """Cleanup PyTorch model."""
        if self.model is not None:
            del self.model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[{self.name}] Cleanup")


class ONNXModelActor(AIModelActor):
    """
    ONNX model actor for cross-framework inference.

    Supports models exported from PyTorch, TensorFlow, etc.
    """

    def __init__(self, model_id: str, bus: MessageBus, config: Config, onnx_path: str):
        super().__init__(model_id, bus, config)
        self.onnx_path = onnx_path
        self.session = None

    def load_model(self) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(self.onnx_path)
            print(f"[{self.name}] ONNX model loaded from {self.onnx_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Load error: {e}")
            return False

    def preprocess(self, input_data: Dict[str, Any]) -> Any:
        """Preprocess for ONNX model."""
        # Convert to numpy arrays with correct dtype
        return input_data

    def infer(self, preprocessed_data: Any) -> Any:
        """Run ONNX inference."""
        input_name = self.session.get_inputs()[0].name
        input_data = preprocessed_data.get('input', preprocessed_data.get('image'))

        outputs = self.session.run(None, {input_name: input_data})
        return outputs

    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Postprocess ONNX output."""
        return {
            'outputs': [o.tolist() for o in model_output],
            'num_outputs': len(model_output)
        }

    def cleanup_model(self) -> None:
        """Cleanup ONNX session."""
        if self.session is not None:
            del self.session
        print(f"[{self.name}] Cleanup")


class PolicyNetworkActor(AIModelActor):
    """
    Policy network actor for robot control.

    Takes robot state + perception as input, outputs actions.
    """

    def __init__(self, model_id: str, bus: MessageBus, config: Config, checkpoint_path: str):
        super().__init__(model_id, bus, config)
        self.checkpoint_path = checkpoint_path

        # Subscribe to robot state and perception for context
        self._robot_state_queue = None
        self._latest_robot_state = None

    def load_model(self) -> bool:
        """Load policy network."""
        try:
            # Load your policy architecture
            print(f"[{self.name}] Policy network loaded from {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"[{self.name}] Load error: {e}")
            return False

    def setup(self) -> None:
        """Setup with additional subscriptions."""
        super().setup()

        # Subscribe to robot state for context
        from ..core.bus import Topics
        self._robot_state_queue = self.bus.subscribe_queue(Topics.ROBOT_STATE, maxsize=2)

    def preprocess(self, input_data: Dict[str, Any]) -> Any:
        """Prepare observation for policy."""
        # Combine robot state + perception + task info
        obs = {
            'joint_positions': input_data.get('joint_positions', []),
            'joint_velocities': input_data.get('joint_velocities', []),
            'image': input_data.get('image', None),
            'task_params': input_data.get('task_params', {})
        }
        return obs

    def infer(self, preprocessed_data: Any) -> Any:
        """Run policy network."""
        # Forward pass through policy
        # action = self.model(obs)
        action = np.random.randn(7)  # Mock: 7-DOF joint action
        return action

    def postprocess(self, model_output: Any) -> Dict[str, Any]:
        """Format action output."""
        return {
            'action': model_output.tolist(),
            'action_type': 'joint_positions'  # or 'joint_torques', 'ee_delta', etc.
        }

    def cleanup_model(self) -> None:
        """Cleanup policy."""
        print(f"[{self.name}] Cleanup")
