#!/usr/bin/env python3
"""
Example 1: Multi-Agent Ethical Decision Workflow

Demonstrates how to use multiple agents to make ethical decisions through
a structured workflow combining ethical analysis, emotional intelligence,
and logical reasoning.

This example shows:
- Sequential agent communication
- Ethical decision-making process
- Multi-stage approval workflow
- Result aggregation
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Import agent utilities
import sys
sys.path.insert(0, '/home/ubuntu/Helix')

from backend.agents import get_agent, AGENTS
from backend.communication import send_message, MessageType, MessagePriority
from backend.services.ucf_analyzer import add_ucf_snapshot, generate_ucf_report


# ============================================================================
# ETHICAL DECISION WORKFLOW
# ============================================================================

class EthicalDecisionWorkflow:
    """Multi-agent workflow for ethical decision-making."""
    
    def __init__(self):
        self.decision_id = None
        self.stages = []
        self.result = None
    
    async def analyze_proposal(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-stage ethical decision analysis.
        
        Stages:
        1. Kael (Ethics) - Ethical analysis
        2. Lumina (Emotional) - Emotional impact assessment
        3. Grok (Analysis) - Logical analysis
        4. Vega (Coordination) - Final decision
        """
        
        self.decision_id = f"decision_{int(datetime.utcnow().timestamp()*1000)}"
        print(f"\n🎯 Starting Ethical Decision Analysis: {self.decision_id}")
        print(f"Proposal: {proposal.get('title', 'Untitled')}")
        print("=" * 70)
        
        # Stage 1: Ethical Analysis (Kael)
        print("\n📋 Stage 1: Ethical Analysis (Kael)")
        print("-" * 70)
        
        kael_analysis = await self._stage_ethical_analysis(proposal)
        self.stages.append({
            'stage': 'ethical_analysis',
            'agent': 'kael',
            'result': kael_analysis
        })
        
        if not kael_analysis.get('passes_ethics', False):
            print(f"❌ Ethical check failed: {kael_analysis.get('reason', 'Unknown')}")
            self.result = {
                'status': 'rejected',
                'reason': 'ethical_violation',
                'details': kael_analysis
            }
            return self.result
        
        print(f"✅ Ethical check passed")
        print(f"   Confidence: {kael_analysis.get('confidence', 0):.1%}")
        print(f"   Concerns: {', '.join(kael_analysis.get('concerns', []))}")
        
        # Stage 2: Emotional Impact Assessment (Lumina)
        print("\n💖 Stage 2: Emotional Impact Assessment (Lumina)")
        print("-" * 70)
        
        lumina_analysis = await self._stage_emotional_assessment(proposal, kael_analysis)
        self.stages.append({
            'stage': 'emotional_assessment',
            'agent': 'lumina',
            'result': lumina_analysis
        })
        
        print(f"✅ Emotional analysis complete")
        print(f"   Impact Score: {lumina_analysis.get('impact_score', 0):.1f}/10")
        print(f"   Sentiment: {lumina_analysis.get('sentiment', 'neutral')}")
        print(f"   Recommendations: {', '.join(lumina_analysis.get('recommendations', []))}")
        
        # Stage 3: Logical Analysis (Grok)
        print("\n🔍 Stage 3: Logical Analysis (Grok)")
        print("-" * 70)
        
        grok_analysis = await self._stage_logical_analysis(proposal, kael_analysis, lumina_analysis)
        self.stages.append({
            'stage': 'logical_analysis',
            'agent': 'grok',
            'result': grok_analysis
        })
        
        print(f"✅ Logical analysis complete")
        print(f"   Feasibility: {grok_analysis.get('feasibility_score', 0):.1%}")
        print(f"   Risk Level: {grok_analysis.get('risk_level', 'unknown')}")
        print(f"   Key Factors: {', '.join(grok_analysis.get('key_factors', [])[:3])}")
        
        # Stage 4: Final Decision (Vega)
        print("\n⚖️  Stage 4: Final Decision (Vega)")
        print("-" * 70)
        
        vega_decision = await self._stage_final_decision(
            proposal,
            kael_analysis,
            lumina_analysis,
            grok_analysis
        )
        self.stages.append({
            'stage': 'final_decision',
            'agent': 'vega',
            'result': vega_decision
        })
        
        self.result = {
            'status': vega_decision.get('decision', 'undecided'),
            'confidence': vega_decision.get('confidence', 0),
            'reasoning': vega_decision.get('reasoning', ''),
            'stages': self.stages
        }
        
        print(f"✅ Final Decision: {vega_decision.get('decision', 'UNDECIDED').upper()}")
        print(f"   Confidence: {vega_decision.get('confidence', 0):.1%}")
        print(f"   Reasoning: {vega_decision.get('reasoning', 'No reasoning provided')}")
        
        return self.result
    
    async def _stage_ethical_analysis(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 1: Ethical analysis by Kael."""
        
        message_content = {
            'proposal': proposal,
            'analysis_type': 'ethical_review'
        }
        
        # Simulate Kael's analysis
        analysis = {
            'passes_ethics': proposal.get('ethical_score', 7) >= 6,
            'confidence': 0.92,
            'concerns': [
                'Potential impact on stakeholders',
                'Long-term consequences'
            ] if proposal.get('ethical_score', 7) < 8 else [],
            'frameworks_applied': ['virtue_ethics', 'consequentialism', 'deontology'],
            'ethical_score': proposal.get('ethical_score', 7),
            'reason': 'Proposal aligns with core ethical principles' if proposal.get('ethical_score', 7) >= 6 else 'Ethical concerns detected'
        }
        
        return analysis
    
    async def _stage_emotional_assessment(
        self,
        proposal: Dict[str, Any],
        ethical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 2: Emotional impact assessment by Lumina."""
        
        # Simulate Lumina's analysis
        assessment = {
            'impact_score': 7.5,
            'sentiment': 'positive',
            'emotional_resonance': 0.85,
            'stakeholder_sentiment': {
                'primary_stakeholders': 'positive',
                'secondary_stakeholders': 'neutral',
                'affected_communities': 'positive'
            },
            'recommendations': [
                'Communicate benefits clearly',
                'Address stakeholder concerns',
                'Build emotional connection'
            ]
        }
        
        return assessment
    
    async def _stage_logical_analysis(
        self,
        proposal: Dict[str, Any],
        ethical_analysis: Dict[str, Any],
        emotional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 3: Logical analysis by Grok."""
        
        # Simulate Grok's analysis
        analysis = {
            'feasibility_score': 0.88,
            'risk_level': 'medium',
            'implementation_complexity': 'moderate',
            'resource_requirements': {
                'time': '3-4 weeks',
                'budget': '$50,000-$75,000',
                'team_size': '5-7 people'
            },
            'key_factors': [
                'Clear implementation path',
                'Manageable risks',
                'Adequate resources available',
                'Timeline is realistic'
            ],
            'success_probability': 0.82
        }
        
        return analysis
    
    async def _stage_final_decision(
        self,
        proposal: Dict[str, Any],
        ethical_analysis: Dict[str, Any],
        emotional_analysis: Dict[str, Any],
        logical_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 4: Final decision by Vega."""
        
        # Aggregate all analyses
        ethical_score = ethical_analysis.get('confidence', 0)
        emotional_score = emotional_analysis.get('emotional_resonance', 0)
        logical_score = logical_analysis.get('success_probability', 0)
        
        overall_score = (ethical_score + emotional_score + logical_score) / 3
        
        # Make decision
        decision = 'approved' if overall_score >= 0.75 else 'rejected' if overall_score < 0.5 else 'needs_review'
        
        vega_decision = {
            'decision': decision,
            'confidence': overall_score,
            'reasoning': f"Proposal evaluated across ethical, emotional, and logical dimensions. Overall score: {overall_score:.1%}",
            'next_steps': [
                'Schedule implementation kickoff',
                'Assign project lead',
                'Begin resource allocation'
            ] if decision == 'approved' else []
        }
        
        return vega_decision
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow summary."""
        return {
            'decision_id': self.decision_id,
            'result': self.result,
            'stages': self.stages,
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run the ethical decision workflow example."""
    
    print("\n" + "="*70)
    print("🌀 HELIX EXAMPLE 1: Multi-Agent Ethical Decision Workflow")
    print("="*70)
    
    # Create workflow
    workflow = EthicalDecisionWorkflow()
    
    # Example proposals to analyze
    proposals = [
        {
            'title': 'Implement AI Transparency Initiative',
            'description': 'Make all AI decision-making processes transparent to users',
            'ethical_score': 9,
            'impact': 'high'
        },
        {
            'title': 'Reduce Safety Testing to Speed Release',
            'description': 'Cut safety testing duration in half to meet market deadline',
            'ethical_score': 3,
            'impact': 'high'
        },
        {
            'title': 'Optimize Resource Allocation',
            'description': 'Redistribute computing resources for better efficiency',
            'ethical_score': 7,
            'impact': 'medium'
        }
    ]
    
    # Analyze each proposal
    for proposal in proposals:
        result = await workflow.analyze_proposal(proposal)
        print("\n" + "-"*70)
        print(f"Summary: {json.dumps(result, indent=2)}")
        print("="*70)
        
        await asyncio.sleep(1)  # Brief pause between proposals
    
    # Display final report
    print("\n" + "="*70)
    print("📊 WORKFLOW ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total proposals analyzed: {len(proposals)}")
    print(f"Approved: {sum(1 for p in proposals if p.get('ethical_score', 5) >= 7)}")
    print(f"Rejected: {sum(1 for p in proposals if p.get('ethical_score', 5) < 4)}")
    print(f"Needs Review: {sum(1 for p in proposals if 4 <= p.get('ethical_score', 5) < 7)}")


if __name__ == "__main__":
    asyncio.run(main())
