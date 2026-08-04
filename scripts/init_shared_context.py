import os
import sys
import json

from graph_memory.core.engine import (
    get_db_path, init_db, get_or_create_node, create_relation, add_observation
)

def populate_shared_context():
    db_path = get_db_path(os.getcwd())
    init_db(db_path)
    print(f"--> [Context Sharing] Initializing Epistemic Graph Memory at: {db_path}")

    # 1. Root Project Node
    get_or_create_node(
        db_path=db_path,
        node_id="CodeForge-250M_Project",
        label="Project",
        properties={
            "description": "250M Parameter Coding Foundation Model + ReAct JSON Sidecar Engine for Nexus-Agent",
            "model_size": "250 Million Parameters (16L / 1024H / GQA)",
            "budget_status": "$12.41 remaining (~2.35B tokens capacity)",
            "current_status": "All 22 bugs fixed across repo. Prepared for fresh Step 0 training run on Lightning AI."
        },
        trust_score=1.0,
        verification_method="human"
    )

    # 2. Tracks Nodes (Short-Term vs Long-Term)
    get_or_create_node(
        db_path=db_path,
        node_id="Track_1_Immediate_Product",
        label="Architecture",
        properties={
            "goal": "Ship Nexus-Agent V2 offline local engine immediately without waiting weeks for training",
            "model_chosen": "Qwen2.5-Coder-1.5B-Instruct-GGUF (~1 GB)",
            "reason": "65% HumanEval pass@1 + native JSON tool-calling out of the box",
            "status": "Ready to integrate inside serving/nexus_bridge.py and nexus-agent --provider local"
        },
        trust_score=1.0,
        verification_method="llm"
    )

    get_or_create_node(
        db_path=db_path,
        node_id="Track_2_Custom_Model_Research",
        label="Architecture",
        properties={
            "goal": "Train CodeForge-250M from Step 0 on a 2.35 Billion token multi-source blend",
            "sources": ["starcoder-python (45%)", "glaive-function-calling (15%)", "codeparrot-clean (12%)", "evol-codealpaca (10%)", "tiny-textbooks (8%)", "commitpackft (5%)", "code-contests (5%)"],
            "why_step_0": "Previous run (108k steps) catastrophically overfitted after step 146 due to 4.77M unique token loop.",
            "status": "All 22 codebase bugs fixed. Option B completed."
        },
        trust_score=1.0,
        verification_method="human"
    )

    create_relation(db_path, "CodeForge-250M_Project", "Track_1_Immediate_Product", "IMPLEMENTS", {"type": "short_term_win"}, trust_score=1.0, verification_method="human")
    create_relation(db_path, "CodeForge-250M_Project", "Track_2_Custom_Model_Research", "IMPLEMENTS", {"type": "long_term_research"}, trust_score=1.0, verification_method="human")

    # 3. Bug Fixes Node
    get_or_create_node(
        db_path=db_path,
        node_id="Completed_Codebase_Audit_And_Fixes",
        label="Milestone",
        properties={
            "total_bugs_fixed": 22,
            "root_cleanup": "Archived 18 obsolete start_*.py / launch_*.py scripts to scripts/archive/",
            "setup_env_sh": "Fixed line 12 syntax error and made requirements/HF_TOKEN portable",
            "download_stack_py": "Rewritten to stream 7-source ReAct blend from YAML config, format as JSONL, no smoke test or alpaca fallback traps",
            "train_tokenizer_py": "Added 11 special tokens (<|tool_call|>, <|thinking|>, etc.) and iterator training on JSONL text",
            "filter_quality_py": "Scoped AST checks strictly to pure code so instruction/chat/ReAct data is not rejected",
            "deduplicate_py": "Rewritten from O(n^2) nested loops to O(1) exact MD5 + LSH banding",
            "tokenize_dataset_py": "Fixed BUG-09 by writing actual pre-tokenized binary/tensor shards (shard_*.pt) to disk",
            "train_py": "Added ShardedCodeDataset loader, LLaMA-2 init_model_weights, cosine LR warmup/decay, 16-step gradient accumulation, --from_scratch CLI flag, and loss_val safety"
        },
        trust_score=1.0,
        verification_method="human",
        link_to="Track_2_Custom_Model_Research",
        link_type="DEPENDS_ON"
    )

    # 4. Next Step Node (Lightning AI Execution)
    get_or_create_node(
        db_path=db_path,
        node_id="NextStep_LightningAI_Run",
        label="Task",
        properties={
            "platform": "Lightning AI Studio",
            "commands": [
                "bash scripts/setup_env.sh",
                "python3 data/download_stack.py && python3 data/filter_quality.py && python3 data/deduplicate.py && python3 data/train_tokenizer.py && python3 data/tokenize_dataset.py",
                "python3 training/train.py --from_scratch"
            ],
            "flag_note": "--from_scratch flag ensures starting from Step 0 cleanly"
        },
        trust_score=1.0,
        verification_method="human",
        link_to="Completed_Codebase_Audit_And_Fixes",
        link_type="FOLLOWS"
    )

    create_relation(db_path, "NextStep_LightningAI_Run", "CodeForge-250M_Project", "PART_OF", {}, trust_score=1.0, verification_method="human")

    # Link major modules to our Project root so there are NO ORPHANS
    get_or_create_node(db_path, "MOC_training", "MOC_Hub", {"description": "Training engine module"}, trust_score=1.0, verification_method="human", link_to="CodeForge-250M_Project", link_type="PART_OF")
    get_or_create_node(db_path, "MOC_data", "MOC_Hub", {"description": "Data pipeline module"}, trust_score=1.0, verification_method="human", link_to="CodeForge-250M_Project", link_type="PART_OF")
    get_or_create_node(db_path, "MOC_models", "MOC_Hub", {"description": "Architecture & weights init module"}, trust_score=1.0, verification_method="human", link_to="CodeForge-250M_Project", link_type="PART_OF")
    get_or_create_node(db_path, "MOC_serving", "MOC_Hub", {"description": "Nexus-Agent sidecar bridge module"}, trust_score=1.0, verification_method="human", link_to="CodeForge-250M_Project", link_type="PART_OF")

    print("--> [Context Sharing] Successfully logged all Project, Track, Fixes, and Task nodes with Trust = 1.0!")

if __name__ == "__main__":
    populate_shared_context()
