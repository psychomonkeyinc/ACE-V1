# c:\ace3\framework.py
# The ACE Master Framework, integrating all 15 CSMs.
# This is a 100% faithful restoration of the original framework logic and data flow.

import torch
import torch.nn as nn
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# Import all 15 CSMs from csms.py
from csms import *

@dataclass
class PerceptionInput:
    """
    Data structure for providing sensory input to the framework.
    This matches the original specification from your files.
    """
    text_input: Optional[str] = None
    audio_features: Optional[torch.Tensor] = None
    visual_features: Optional[torch.Tensor] = None
    consciousness_context: Optional[torch.Tensor] = None
    timestamp: float = 0.0
    user_id: Optional[str] = None
    context_metadata: Optional[Dict] = None

@dataclass
class ConsciousnessState:
    """
    Restored from your original framework to manage internal state.
    """
    consciousness_level: float = 0.1
    emotions: Dict[str, float] = field(default_factory=lambda: {
        'joy': 0.3, 'curiosity': 0.8, 'empathy': 0.7,
        'anxiety': 0.2, 'nurturing': 0.6
    })
    vocal_development_stage: VocalizationType = VocalizationType.RANDOM_SOUND
    vocal_progress: float = 0.1
    memory_active: bool = True
    relationship_bonds: Dict[str, Any] = field(default_factory=dict)
    current_focus: str = 'initialization'

class ACEMasterFramework(nn.Module):
    """
    The complete ACE Master Framework, containing and orchestrating the 
    full 15-CSM architecture, restored to its original design.
    """
    def __init__(self, 
                 input_dim: int = 2048,
                 unified_dim: int = 1024,
                 vocab_size: int = 50000):
        super().__init__()
        
        self.input_dim = input_dim
        self.unified_dim = unified_dim
        self.vocab_size = vocab_size
        
        # Initialize all 15 CSMs from csms.py
        self.conscience_router = ConsciousnessRouter(input_dim, unified_dim)
        self.perception_csm = PerceptionCSM(unified_dim)
        self.language_csm = LanguageCSM(vocab_size, unified_dim)
        self.memory_csm = MemoryCSM(unified_dim)
        self.cognitive_csm = CognitiveCSM(unified_dim)
        self.output_csm = OutputCSM(unified_dim, vocab_size)
        self.learning_csm = LearningCSM(unified_dim)
        self.security_csm = SecurityCSM(unified_dim)
        self.health_csm = HealthCSM(unified_dim)
        self.vocal_csm = VocalCSM(unified_dim)
        self.emotion_simulation_csm = EmotionSimulationCSM(unified_dim)
        self.theory_of_mind_csm = TheoryOfMindCSM(unified_dim)
        self.emotional_memory_csm = EmotionalMemoryCSM(unified_dim)
        self.attachment_style_csm = AttachmentStyleCSM(unified_dim)
        self.relationship_memory_csm = RelationshipMemoryCSM(unified_dim)
        
        # System state, restored from original
        self.system_state = ConsciousnessState()
        
        print("✅ ACEMasterFramework initialized with full 15-CSM architecture (Original Logic).")

    async def process_consciousness_cycle(self, perception_input: PerceptionInput) -> Dict[str, Any]:
        """
        Processes a complete consciousness cycle, restoring the original data flow
        and inter-module connections from your design.
        """
        try:
            device = next(self.parameters()).device
            # Ensure all inputs are valid tensors on the correct device
            if perception_input.audio_features is None:
                perception_input.audio_features = torch.randn(1, self.unified_dim, dtype=torch.float16, device=device)
            if perception_input.visual_features is None:
                perception_input.visual_features = torch.randn(1, self.unified_dim, dtype=torch.float16, device=device)
            if perception_input.consciousness_context is None:
                perception_input.consciousness_context = torch.randn(1, self.unified_dim, dtype=torch.float16, device=device)
                
            # Phase 1: Perception processing
            perception_outputs = self.perception_csm(
                perception_input.audio_features,
                perception_input.visual_features
            )
            
            # Phase 2: Language processing
            language_outputs = self.language_csm(
                perception_input,
                perception_input.consciousness_context
            )
            
            # Phase 3: Memory consolidation
            memory_outputs = self.memory_csm(
                perception_outputs["fused_features"]
            )
            
            # Phase 4: Cognitive processing
            cognitive_outputs = self.cognitive_csm(
                language_outputs["global_repr"],
                memory_outputs["consolidated_memory"],
                perception_outputs["fused_features"]
            )
            
            # Phase 5: Output generation
            output_results = self.output_csm(
                cognitive_outputs["unified_cognitive_state"]
            )
            
            # Phase 6: Learning adaptation
            learning_results = self.learning_csm(
                cognitive_outputs["unified_cognitive_state"]
            )
            
            # Phase 7: Security analysis
            security_results = self.security_csm(
                perception_outputs["fused_features"]
            )
            
            # Phase 8: Health monitoring
            health_results = self.health_csm(
                cognitive_outputs["unified_cognitive_state"]
            )
            
            # Phase 9: Vocal development
            vocal_results = self.vocal_csm(
                cognitive_outputs["unified_cognitive_state"]
            )
            
            # Phase 10-15: Additional CSMs, following the original logic
            emotion_results = self.emotion_simulation_csm(perception_outputs["fused_features"])
            tom_results = self.theory_of_mind_csm(cognitive_outputs["unified_cognitive_state"])
            emotional_memory_results = self.emotional_memory_csm(emotion_results["emotional_state"])
            attachment_results = self.attachment_style_csm(tom_results["empathic_understanding"]["empathy_features"])
            relationship_results = self.relationship_memory_csm(attachment_results["attachment_security"])
            
            # Conscience router coordination
            conscience_results = self.conscience_router(cognitive_outputs["unified_cognitive_state"])
            
            # Compile the complete results dictionary, as in your original file
            complete_results = {
                "perception": perception_outputs,
                "language": language_outputs,
                "memory": memory_outputs,
                "cognitive": cognitive_outputs,
                "output": output_results,
                "learning": learning_results,
                "security": security_results,
                "health": health_results,
                "vocal": vocal_results,
                "emotion_simulation": emotion_results,
                "theory_of_mind": tom_results,
                "emotional_memory": emotional_memory_results,
                "attachment_style": attachment_results,
                "relationship_memory": relationship_results,
                "conscience": conscience_results,
                "system_metrics": {
                    "consciousness_level": self.system_state.consciousness_level,
                    "processing_timestamp": time.time()
                }
            }
            
            return complete_results
            
        except Exception as e:
            print(f"❌ Consciousness cycle error: {str(e)}")
            return {
                "error": f"Consciousness cycle error: {str(e)}",
                "system_metrics": {
                    "consciousness_level": 0.0,
                    "processing_timestamp": time.time()
                }
            }