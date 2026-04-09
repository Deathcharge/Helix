#!/usr/bin/env python3
"""
Example 3: Parallel Agent Consensus Decision-Making

Demonstrates how to query multiple agents in parallel to reach consensus
on complex decisions. Shows parallel message handling, response aggregation,
and consensus scoring.

This example shows:
- Parallel agent queries
- Response aggregation
- Consensus scoring
- Confidence calculation
- Decision synthesis
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, '/home/ubuntu/Helix')

from backend.communication import send_message, MessageType, MessagePriority


# ============================================================================
# CONSENSUS DECISION MAKER
# ============================================================================

class ConsensusDecisionMaker:
    """Makes decisions through parallel agent consensus."""
    
    def __init__(self):
        self.decision_id = None
        self.responses = {}
        self.consensus_result = None
    
    async def reach_consensus(
        self,
        topic: str,
        context: Dict[str, Any],
        agent_list: List[str] = None
    ) -> Dict[str, Any]:
        """
        Reach consensus on a topic by querying multiple agents in parallel.
        
        Args:
            topic: Decision topic
            context: Context information
            agent_list: List of agents to query (None = default set)
        
        Returns:
            Consensus decision with confidence scores
        """
        
        self.decision_id = f"consensus_{int(datetime.utcnow().timestamp()*1000)}"
        
        if agent_list is None:
            agent_list = self._get_default_agent_list()
        
        print("\n" + "="*70)
        print("🎯 PARALLEL CONSENSUS DECISION-MAKING")
        print("="*70)
        print(f"Topic: {topic}")
        print(f"Agents: {', '.join(agent_list)}")
        print(f"Decision ID: {self.decision_id}")
        print("-"*70)
        
        # Query all agents in parallel
        print("\n📤 Querying agents in parallel...")
        responses = await self._query_agents_parallel(topic, context, agent_list)
        
        print(f"✅ Received {len(responses)} responses")
        
        # Aggregate responses
        print("\n📊 Aggregating responses...")
        aggregated = self._aggregate_responses(responses, agent_list)
        
        # Calculate consensus
        print("\n🔄 Calculating consensus...")
        consensus = self._calculate_consensus(aggregated)
        
        self.consensus_result = consensus
        
        return consensus
    
    async def _query_agents_parallel(
        self,
        topic: str,
        context: Dict[str, Any],
        agent_list: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Query all agents in parallel."""
        
        tasks = []
        for agent_id in agent_list:
            task = self._query_single_agent(topic, context, agent_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        responses = {}
        for agent_id, result in zip(agent_list, results):
            if isinstance(result, Exception):
                responses[agent_id] = {
                    'status': 'error',
                    'error': str(result)
                }
            else:
                responses[agent_id] = result
        
        self.responses = responses
        return responses
    
    async def _query_single_agent(
        self,
        topic: str,
        context: Dict[str, Any],
        agent_id: str
    ) -> Dict[str, Any]:
        """Query a single agent."""
        
        # Simulate agent query with delay
        await asyncio.sleep(0.1)
        
        # Generate response based on agent type
        response = self._generate_agent_response(agent_id, topic, context)
        
        print(f"  ✓ {agent_id}: {response.get('recommendation', 'no_recommendation')}")
        
        return response
    
    def _generate_agent_response(
        self,
        agent_id: str,
        topic: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate simulated agent response."""
        
        # Map agents to their perspectives
        agent_perspectives = {
            'kael': {
                'perspective': 'ethical',
                'recommendation': 'approve' if context.get('ethical_score', 5) >= 7 else 'reject',
                'confidence': 0.92,
                'reasoning': 'Ethical alignment is strong' if context.get('ethical_score', 5) >= 7 else 'Ethical concerns present'
            },
            'lumina': {
                'perspective': 'emotional',
                'recommendation': 'approve' if context.get('impact_score', 5) >= 6 else 'review',
                'confidence': 0.88,
                'reasoning': 'Positive emotional resonance detected'
            },
            'grok': {
                'perspective': 'analytical',
                'recommendation': 'approve' if context.get('feasibility', 0.7) >= 0.75 else 'review',
                'confidence': 0.85,
                'reasoning': 'Feasibility analysis shows good prospects'
            },
            'oracle': {
                'perspective': 'predictive',
                'recommendation': 'approve' if context.get('success_rate', 0.7) >= 0.8 else 'review',
                'confidence': 0.80,
                'reasoning': 'Predictive models show positive outcomes'
            },
            'shadow': {
                'perspective': 'psychological',
                'recommendation': 'approve' if context.get('psychological_safety', 0.7) >= 0.7 else 'review',
                'confidence': 0.82,
                'reasoning': 'Psychological impact assessment positive'
            },
            'aether': {
                'perspective': 'quantum',
                'recommendation': 'approve' if context.get('complexity', 0.5) <= 0.8 else 'review',
                'confidence': 0.78,
                'reasoning': 'Quantum analysis indicates feasibility'
            }
        }
        
        return agent_perspectives.get(agent_id, {
            'perspective': 'unknown',
            'recommendation': 'abstain',
            'confidence': 0.5,
            'reasoning': 'Unable to analyze'
        })
    
    def _aggregate_responses(
        self,
        responses: Dict[str, Dict[str, Any]],
        agent_list: List[str]
    ) -> Dict[str, Any]:
        """Aggregate responses from all agents."""
        
        print("\n  Response Summary:")
        print("  " + "-"*66)
        
        recommendations = {
            'approve': 0,
            'reject': 0,
            'review': 0,
            'abstain': 0
        }
        
        confidences = []
        perspectives = []
        
        for agent_id, response in responses.items():
            if response.get('status') == 'error':
                print(f"  {agent_id:15} | ERROR: {response.get('error', 'Unknown error')}")
                continue
            
            rec = response.get('recommendation', 'abstain')
            conf = response.get('confidence', 0)
            persp = response.get('perspective', 'unknown')
            
            recommendations[rec] += 1
            confidences.append(conf)
            perspectives.append(persp)
            
            print(f"  {agent_id:15} | {rec:10} | Confidence: {conf:.0%} | {persp}")
        
        return {
            'recommendations': recommendations,
            'confidences': confidences,
            'perspectives': perspectives,
            'total_responses': len(responses)
        }
    
    def _calculate_consensus(self, aggregated: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus from aggregated responses."""
        
        recommendations = aggregated['recommendations']
        confidences = aggregated['confidences']
        
        # Determine primary recommendation
        primary_rec = max(recommendations.items(), key=lambda x: x[1])[0]
        primary_count = recommendations[primary_rec]
        
        # Calculate consensus strength
        total_votes = sum(recommendations.values())
        consensus_strength = primary_count / total_votes if total_votes > 0 else 0
        
        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Determine consensus level
        if consensus_strength >= 0.8:
            consensus_level = 'strong'
        elif consensus_strength >= 0.6:
            consensus_level = 'moderate'
        elif consensus_strength >= 0.4:
            consensus_level = 'weak'
        else:
            consensus_level = 'no_consensus'
        
        print("\n  Consensus Calculation:")
        print("  " + "-"*66)
        print(f"  Primary Recommendation: {primary_rec.upper()}")
        print(f"  Votes: {primary_count}/{total_votes}")
        print(f"  Consensus Strength: {consensus_strength:.0%}")
        print(f"  Consensus Level: {consensus_level.upper()}")
        print(f"  Average Confidence: {avg_confidence:.0%}")
        
        # Vote breakdown
        print("\n  Vote Breakdown:")
        for rec, count in recommendations.items():
            pct = (count / total_votes * 100) if total_votes > 0 else 0
            print(f"    {rec:10}: {count} votes ({pct:.0f}%)")
        
        consensus = {
            'decision_id': self.decision_id,
            'primary_recommendation': primary_rec,
            'consensus_strength': consensus_strength,
            'consensus_level': consensus_level,
            'average_confidence': avg_confidence,
            'vote_breakdown': recommendations,
            'recommendation_rationale': self._generate_rationale(
                primary_rec,
                consensus_strength,
                aggregated['perspectives']
            ),
            'confidence_scores': confidences,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return consensus
    
    def _generate_rationale(
        self,
        recommendation: str,
        strength: float,
        perspectives: List[str]
    ) -> str:
        """Generate rationale for the consensus decision."""
        
        strength_desc = {
            'strong': 'overwhelming',
            'moderate': 'clear',
            'weak': 'slight',
            'no_consensus': 'no'
        }
        
        strength_level = 'strong' if strength >= 0.8 else 'moderate' if strength >= 0.6 else 'weak' if strength >= 0.4 else 'no_consensus'
        strength_word = strength_desc.get(strength_level, 'unclear')
        
        rationale = f"The agent collective reached {strength_word} consensus to {recommendation.upper()} this proposal. "
        rationale += f"Analysis from {len(set(perspectives))} distinct perspectives supports this decision with {strength:.0%} agreement."
        
        return rationale
    
    def _get_default_agent_list(self) -> List[str]:
        """Get default list of agents for consensus."""
        return [
            'kael',      # Ethics
            'lumina',    # Emotional
            'grok',      # Analysis
            'oracle',    # Prediction
            'shadow',    # Psychology
            'aether'     # Quantum
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get consensus summary."""
        return {
            'decision_id': self.decision_id,
            'result': self.consensus_result,
            'responses': self.responses,
            'timestamp': datetime.utcnow().isoformat()
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Run the parallel consensus example."""
    
    print("\n" + "="*70)
    print("🌀 HELIX EXAMPLE 3: Parallel Agent Consensus Decision-Making")
    print("="*70)
    
    # Create decision maker
    maker = ConsensusDecisionMaker()
    
    # Example decisions to reach consensus on
    decisions = [
        {
            'topic': 'Implement Advanced AI Safety Measures',
            'context': {
                'ethical_score': 9,
                'impact_score': 8,
                'feasibility': 0.85,
                'success_rate': 0.88,
                'psychological_safety': 0.9,
                'complexity': 0.6
            }
        },
        {
            'topic': 'Accelerate Release Timeline',
            'context': {
                'ethical_score': 5,
                'impact_score': 6,
                'feasibility': 0.65,
                'success_rate': 0.60,
                'psychological_safety': 0.5,
                'complexity': 0.8
            }
        },
        {
            'topic': 'Expand Agent Autonomy',
            'context': {
                'ethical_score': 7,
                'impact_score': 7,
                'feasibility': 0.75,
                'success_rate': 0.78,
                'psychological_safety': 0.72,
                'complexity': 0.7
            }
        }
    ]
    
    # Process each decision
    for i, decision in enumerate(decisions, 1):
        print(f"\n{'='*70}")
        print(f"DECISION {i}/{len(decisions)}")
        print(f"{'='*70}")
        
        consensus = await maker.reach_consensus(
            topic=decision['topic'],
            context=decision['context']
        )
        
        print("\n" + "-"*70)
        print("FINAL CONSENSUS RESULT")
        print("-"*70)
        print(f"Recommendation: {consensus['primary_recommendation'].upper()}")
        print(f"Confidence: {consensus['average_confidence']:.0%}")
        print(f"Consensus Level: {consensus['consensus_level'].upper()}")
        print(f"Rationale: {consensus['recommendation_rationale']}")
        
        if i < len(decisions):
            await asyncio.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("📊 CONSENSUS ANALYSIS COMPLETE")
    print("="*70)
    print(f"Total decisions analyzed: {len(decisions)}")
    print(f"Average consensus strength: {sum(d.get('consensus_strength', 0) for d in [maker.consensus_result]) / 1:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
