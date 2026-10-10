import json
import warnings
import re
from typing import List, Tuple, Optional
from PIL import Image
from pydantic import BaseModel, field_validator
from .vlm_chat import VlmChat
from src.llm2pddl.domains import Domain
from src.llm2pddl.problem import Problem


def _normalize_and_deduplicate(v):
    if not isinstance(v, list):
        return v
    import ast
    seen = set()
    out = []
    for x in v:
        if isinstance(x, dict):
            items = list(x.values())
        else:
            clean = str(x).strip()
            if clean.startswith("{") and clean.endswith("}"):
                try:
                    parsed = ast.literal_eval(clean)
                    items = list(parsed.values()) if isinstance(parsed, dict) else [clean]
                except Exception:
                    items = [clean]
            else:
                items = [clean]
        for item in items:
            clean_item = str(item).strip().strip("'\"")
            norm = clean_item.lower().replace(" ", "_")
            if norm and norm not in seen:
                seen.add(norm)
                out.append(clean_item)
    return out


def _merge_clarified_entities(new_items: List[str], prev_items: List[str]) -> List[str]:
    """
    Intelligently merges new items from clarification with previously identified items:
    1. Preserves newly identified/refined items from new_items.
    2. Keeps un-refined items from prev_items that were not mentioned or replaced.
    3. Drops generic items from prev_items if a more specific version is in new_items (e.g. 'can' replaced by 'red can').
    """
    merged = list(_normalize_and_deduplicate(new_items))
    if not prev_items:
        return merged
    if not merged:
        return list(_normalize_and_deduplicate(prev_items))

    for prev in prev_items:
        p_clean = prev.strip().lower()
        # Check if already present
        if any(p_clean == m.strip().lower() for m in merged):
            continue
        # Check if prev was a generic form that is now refined in merged (e.g. 'can' -> 'red can')
        is_refined = any(p_clean in m.strip().lower().split() for m in merged)
        if not is_refined:
            merged.append(prev)
    return _normalize_and_deduplicate(merged)


class ObjGrounding(BaseModel):
    """
    Schema for manipulable objects that the robot picks up, moves, or holds.
    """
    task_objects: List[str]

    @field_validator("task_objects", mode="before")
    @classmethod
    def deduplicate(cls, v):
        return _normalize_and_deduplicate(v)


class LocGrounding(BaseModel):
    """
    Schema for target locations, receptacles, or drop zones.
    """
    target_locations: List[str]

    @field_validator("target_locations", mode="before")
    @classmethod
    def deduplicate(cls, v):
        return _normalize_and_deduplicate(v)


class SceneGrounding(BaseModel):
    """
    Unified schema of grounded scene entities, separating manipulable objects from target locations.
    """
    task_objects: List[str]
    target_locations: List[str]

    @field_validator("task_objects", "target_locations", mode="before")
    @classmethod
    def deduplicate(cls, v):
        return _normalize_and_deduplicate(v)


class AmbiguityReasoning(BaseModel):
    task_ambiguous: bool
    explanation: str
    clarifying_question: str


class AmbresStructured(VlmChat):
    def __init__(self, vlm_name="qwen2.5vl:3b"):
        super().__init__(vlm_name=vlm_name)

    def add_message(self, role: str, content:str):
        super().add_message(role, content)

    def handle_query(self, task_description: str, image) -> dict:
        """
        Handle the query from the user. Accepts a single PIL image or a list of PIL images.
        """
        # set the image(s) and construct the message for the vlm query
        if isinstance(image, list):
            self.set_image(image)
            img_desc = f"Please analyze the attached {len(image)} images and extract all relevant entities across all actions. "
        else:
            self.set_image([image])
            img_desc = "Please analyze the attached image and extract all relevant entities across all actions. "

        user_text = (
            "Task Description: " + task_description + "\n" +
            img_desc +
            "Carefully examine the entire task description from start to finish.\n"
            "Return a strictly valid JSON object with the following schema:\n"
            "{\n"
            '  "task_objects": [...],\n'
            '  "target_locations": [...]\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. 'task_objects': ONLY manipulable items the robot must pick up or move (e.g. cans, cubes, fruits, mugs). Include EVERY object mentioned across ALL steps!\n"
            "2. 'target_locations': ONLY receptacles, containers, or destinations (e.g. sorting bin, pot, bowl, drawer, table) where items are placed.\n"
            "3. Keep all adjectives and colors exactly as written in the task description (e.g., 'red can', 'yellow cube', 'blue can').\n"
            "4. Do not omit any object or location mentioned in any step!"
        )
        self.add_message("user", user_text)
        text_out = self.inference(
            self.messages,
            do_sample=False,
            format_schema="json"
        )
        # get ai reply and save to self.messages
        self.add_message("assistant", text_out)
        try:
            grounded_json = json.loads(self.clean_json(text_out))
            task_objects = grounded_json.get("task_objects", [])
            target_locations = grounded_json.get("target_locations", [])
        except Exception:
            warnings.warn("Failed to parse VLM scene grounding JSON. Falling back to defaults.")
            task_objects = []
            target_locations = []

        # Store last groundings for safe multi-turn clarification consolidation
        self.last_task_objects = _normalize_and_deduplicate(task_objects)
        self.last_target_locations = _normalize_and_deduplicate(target_locations)

        # ambiguity resolution
        ambig_query = (
            "Analyze whether the task description is ambiguous given the visible objects and receptacles in the scene.\n"
            "A task is AMBIGUOUS (task_ambiguous = true) if:\n"
            "1. An extracted object or location is generic (e.g. 'can', 'cube', 'bin', 'box') and there are multiple candidate items of that type visible.\n"
            "2. An item's color, size, or identity is underspecified so the robot cannot uniquely decide which one to grasp or place into.\n"
            "If ambiguous, provide a clear, specific 'clarifying_question' asking the user to specify which entity they want.\n"
            "If unambiguous, set task_ambiguous = false and clarifying_question = \"\".\n\n"
            "Return strictly a JSON object with:\n"
            "{\n"
            '  "task_ambiguous": true/false,\n'
            '  "explanation": "...",\n'
            '  "clarifying_question": "..."\n'
            "}"
        )
        self.add_message("user", ambig_query)
        text_out = self.inference(
            self.messages,
            do_sample=False,
            format_schema="json"
        )
        # get ai reply (claryfing question) and save to self.messages
        self.add_message("assistant", text_out)

        try:
            data_out = json.loads(self.clean_json(text_out))
            if not isinstance(data_out, dict):
                data_out = {"task_ambiguous": False, "clarifying_question": ""}
        except Exception:
            warnings.warn("LLM Output broken, will output default response.")
            data_out = {"task_ambiguous": False, "clarifying_question": ""}
        task_ambiguous = bool(data_out.get("task_ambiguous", False))

        clarifying_question = data_out.get("clarifying_question", "") if task_ambiguous else ""
        output = {
            "task_objects": self.last_task_objects,
            "target_locations": self.last_target_locations,
            "task_ambiguous": task_ambiguous,
            "clarifying_question": clarifying_question,
        }
        return output

    def handle_query_dict(self, input_data: dict) -> dict:
        task_description: str = input_data["task_description"]
        image = Image.open(input_data["image_path"])
        # Maintain high resolution for accurate perception (only scale if exceeding 1024)
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        output = self.handle_query(task_description, image)
        return output

    def handle_response(self, response: str) -> dict:
        """
        Handle user clarification response and consolidate with previously confirmed entities.
        """
        clarification_msg = (
            f"User Clarification: {response}\n\n"
            "Based on the original task, the image, and this clarification, provide the FINAL, COMPLETE JSON "
            "containing ALL grounded entities across the entire task:\n"
            "{\n"
            '  "task_objects": [...],\n'
            '  "target_locations": [...]\n'
            "}"
        )
        self.add_message("user", clarification_msg)
        text_out = self.inference(
            self.messages,
            do_sample=False,
            format_schema="json"
        )
        self.add_message("assistant", text_out)
        try:
            grounded_json = json.loads(self.clean_json(text_out))
            task_objects = grounded_json.get("task_objects", [])
            target_locations = grounded_json.get("target_locations", [])
        except Exception:
            task_objects = []
            target_locations = []

        norm_objs = _normalize_and_deduplicate(task_objects)
        norm_locs = _normalize_and_deduplicate(target_locations)

        prev_objs = getattr(self, "last_task_objects", [])
        prev_locs = getattr(self, "last_target_locations", [])

        # Intelligently merge without dropping previously confirmed objects or receptacles
        final_objs = _merge_clarified_entities(norm_objs, prev_objs)
        final_locs = _merge_clarified_entities(norm_locs, prev_locs)

        self.last_task_objects = final_objs
        self.last_target_locations = final_locs

        output = {
            "task_objects": final_objs,
            "target_locations": final_locs,
        }
        return output

    def generate_problem_json(
        self, 
        initial_state_desc: str, 
        task_description: str, 
        task_objects: List[str], 
        domain_pddl: str,
        target_locations: Optional[List[str]] = None
    ) -> Problem:
        """
        Generates Problem JSON object given a predefined PDDL domain definition.
        """
        json_system_prompt = (
            "You are an expert PDDL generator. Output ONLY a single valid JSON object. "
            "Do not include any conversational text, markdown notes, or explanations outside the JSON.\n\n"
            "The JSON must follow this exact structure:\n"
            "{\n"
            '  "problem": {\n'
            '    "problem_name": "...",\n'
            '    "domain_name": "...",\n'
            '    "objects": [{"name": "...", "type": "..."}],\n'
            '    "init": [{"name": "...", "parameters": ["..."]}],\n'
            '    "goal": [{"name": "...", "parameters": ["..."]}]\n'
            '  }\n'
            "}"
        )

        few_shot_user = (
            "Domain PDDL:\n"
            "(define (domain manipulation)\n"
            "  (:requirements :strips :typing)\n"
            "  (:types robot obj location)\n"
            "  (:predicates (holding ?ob) (on-table ?ob) (on ?ob ?pos))\n"
            "  (:action pick :parameters (?ob - obj) :precondition (and (not (holding ?ob)) (on-table ?ob)) :effect (and (holding ?ob) (not (on-table ?ob))))\n"
            "  (:action place :parameters (?ob - obj ?pos - location) :precondition (and (holding ?ob)) :effect (and (not (holding ?ob)) (on ?ob ?pos)))\n"
            ")\n\n"
            "Initial State Description: Both can_A and cube_B are resting on the table. The robot's arm is empty.\n"
            "Goal Task: Place can_A and cube_B into the bin.\n"
            "Grounded Objects: [\"can_A\", \"cube_B\"]\n"
            "Target Locations: [\"bin\"]\n"
            "Generate the PDDL Problem JSON object."
        )

        few_shot_assistant = """{
            "problem": {
                "problem_name": "manipulation-task",
                "domain_name": "manipulation",
                "objects": [
                {"name": "can_A", "type": "obj"},
                {"name": "cube_B", "type": "obj"},
                {"name": "bin", "type": "location"}
                ],
                "init": [
                {"name": "on-table", "parameters": ["can_A"]},
                {"name": "on-table", "parameters": ["cube_B"]}
                ],
                "goal": [
                {"name": "on", "parameters": ["can_A", "bin"]},
                {"name": "on", "parameters": ["cube_B", "bin"]}
                ]
            }
            }"""

        # Sanitize target locations to valid PDDL identifiers (underscores, no spaces)
        safe_locations = [loc.strip().replace(" ", "_") for loc in (target_locations or [])]

        # Convert spaces to underscores in task objects and ensure strict type separation from locations
        safe_task_objects = [
            obj.strip().replace(" ", "_")
            for obj in task_objects
            if obj.strip().replace(" ", "_") not in safe_locations
        ]

        user_prompt = (
            f"Domain PDDL:\n{domain_pddl}\n\n"
            f"Initial State Description: {initial_state_desc}\n"
            f"Goal Task: {task_description}\n"
            f"Grounded Manipulable Objects (type 'obj'): {json.dumps(safe_task_objects)}\n"
            f"Target Locations / Receptacles (type 'location'): {json.dumps(safe_locations)}\n"
            "You are an expert PDDL generator. You must return a single valid JSON object with a single top-level key: 'problem'.\n\n"
            "Output strictly valid JSON. Always use double quotes for all dictionary keys and string values. Never use single quotes (').\n"
            "CRITICAL RULES:\n"
            "1. Every single entity appearing in ':init' or ':goal' MUST be declared in the 'objects' list.\n"
            "2. Manipulable items (e.g. red_can, blue_can, yellow_cube) MUST be declared in 'objects' with type 'obj'.\n"
            "3. Target receptacles and containers (e.g. sorting_bin, pot, bin, bowl) MUST be declared in 'objects' with type 'location'. Do NOT use the type word 'location' as an object name.\n"
            "4. ONLY use predicates explicitly defined in the provided Domain PDDL. If a relationship like 'in' is requested, map it to 'on' with arguments: (on <obj> <location>).\n"
            "5. Ensure all object and location names use underscores instead of spaces (e.g., 'red_can', 'sorting_bin').\n"
            "6. Make sure the 'domain_name' matches the name in the Domain PDDL exactly.\n"
            "7. The ':goal' block must ONLY describe the FINAL physical state after all tasks are completed. Do NOT include intermediate action steps like 'holding' in the goal.\n\n"
            f"Schema for Problem: {Problem.model_json_schema()}"
        )

        messages = [
            {"role": "system", "content": json_system_prompt},
            {"role": "user", "content": few_shot_user},
            {"role": "assistant", "content": few_shot_assistant},
            {"role": "user", "content": user_prompt}
        ]

        problem_obj = self.generate_problem_with_retry(messages, domain_pddl)
        
        return problem_obj

    def generate_problem_with_retry(self, messages: list, domain_pddl: str, max_retries: int = 5):
        # Extract valid predicates and parameter signatures from domain_pddl string
        def parse_pddl_typed_params(param_str):
            tokens = param_str.strip().split()
            types = []
            pending_vars = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok == '-':
                    if i + 1 < len(tokens):
                        type_name = tokens[i + 1]
                        types.extend([type_name] * len(pending_vars))
                        pending_vars = []
                        i += 2
                        continue
                elif tok.startswith('?'):
                    pending_vars.append(tok)
                i += 1
            types.extend(['object'] * len(pending_vars))
            return types

        valid_predicates = {"="}
        predicate_signatures = {}
        pred_match = re.search(r'\(:predicates(.*?)(?:\(:|\)$)', domain_pddl, re.DOTALL | re.IGNORECASE)
        if pred_match:
            section = pred_match.group(1)
            pred_decls = re.findall(r'\(\s*([a-zA-Z0-9_\-]+)(.*?)\)', section, re.DOTALL)
            for pred_name, params_str in pred_decls:
                valid_predicates.add(pred_name)
                predicate_signatures[pred_name] = parse_pddl_typed_params(params_str)

        # Fallback / enhancement: infer predicate parameter types from :action signatures if untyped in (:predicates)
        action_matches = re.finditer(r'\(:action\s+[a-zA-Z0-9_\-]+\s+:parameters\s*\((.*?)\)(.*?)(?=\(:action|\)\s*$)', domain_pddl, re.DOTALL | re.IGNORECASE)
        for act in action_matches:
            param_str = act.group(1)
            body_str = act.group(2)
            tokens = param_str.strip().split()
            var_to_type = {}
            pending = []
            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok == '-':
                    if i + 1 < len(tokens):
                        t = tokens[i + 1]
                        for v in pending:
                            var_to_type[v] = t
                        pending = []
                        i += 2
                        continue
                elif tok.startswith('?'):
                    pending.append(tok)
                i += 1
            for v in pending:
                var_to_type[v] = 'object'

            clean_body = re.sub(r'\b(not|and|or)\b', '', body_str)
            pred_calls = re.findall(r'\(\s*([a-zA-Z0-9_\-]+)([^()]*)\)', clean_body)
            for p_name, args_str in pred_calls:
                args = [a for a in args_str.split() if a.startswith('?')]
                if not args:
                    continue
                inferred = [var_to_type.get(a, 'object') for a in args]
                if p_name in predicate_signatures:
                    current = predicate_signatures[p_name]
                    if not current or all(t == 'object' for t in current):
                        predicate_signatures[p_name] = inferred
                else:
                    predicate_signatures[p_name] = inferred

        # Extract valid types from domain_pddl string
        valid_types = set() 
        type_match = re.search(r'\(:types(.*?)(?:\(:|\)$)', domain_pddl, re.DOTALL | re.IGNORECASE)
        if type_match:
            types_found = re.findall(r'([a-zA-Z0-9_\-]+)', type_match.group(1))
            valid_types.update(types_found)
        
        # If no types were explicitly declared, allow "object" as the fallback
        if not valid_types:
            valid_types.add("object")

        for attempt in range(max_retries):
            # 1. Generate text from LLM
            text_out = self.inference(messages)
            
            try:
                # 2. Parse and validate
                cleaned_json = json.loads(self.clean_json(text_out))
                problem_obj = Problem.model_validate(cleaned_json["problem"])
                
                # Auto-heal: If an undeclared entity is used in :init or :goal, auto-declare it with inferred type
                declared_objects = {obj.name for obj in problem_obj.objects}
                from src.llm2pddl.problem import PDDLObject
                for pred in list(problem_obj.init) + list(problem_obj.goal):
                    expected_param_types = predicate_signatures.get(pred.name, [])
                    for idx, param in enumerate(pred.parameters):
                        clean_p = param.strip().replace(" ", "_")
                        if clean_p and clean_p not in declared_objects and clean_p not in valid_types:
                            # Infer entity type from the predicate's parameter declaration order
                            if idx < len(expected_param_types) and expected_param_types[idx] in valid_types:
                                inferred_type = expected_param_types[idx]
                            else:
                                inferred_type = "obj" if "obj" in valid_types else "object"
                            problem_obj.objects.append(PDDLObject(name=clean_p, type=inferred_type))
                            declared_objects.add(clean_p)

                # Validate predicates, types, and undeclared objects
                errors = []
                for obj in problem_obj.objects:
                    if obj.type not in valid_types:
                        errors.append(f"Hallucinated type for object '{obj.name}' -> '{obj.type}' is not in Domain types! Valid types are: {list(valid_types)}")
                for pred in problem_obj.init:
                    if pred.name not in valid_predicates:
                        errors.append(f"Hallucinated predicate in :init -> '{pred.name}' is not in Domain predicates! Valid predicates are: {list(valid_predicates)}")
                    for param in pred.parameters:
                        if param not in declared_objects:
                            errors.append(f"Undeclared object in :init -> '{param}' used in predicate '{pred.name}' but missing from :objects list!")
                for pred in problem_obj.goal:
                    if pred.name not in valid_predicates:
                        errors.append(f"Hallucinated predicate in :goal -> '{pred.name}' is not in Domain predicates! Valid predicates are: {list(valid_predicates)}")
                    for param in pred.parameters:
                        if param not in declared_objects:
                            errors.append(f"Undeclared object in :goal -> '{param}' used in predicate '{pred.name}' but missing from :objects list!")
                
                if errors:
                    raise ValueError("PDDL Validation Failed:\n" + "\n".join(errors))
                
                # If everything passes, return the valid object
                return problem_obj
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed with error: {e}")
                if attempt == max_retries - 1:
                    raise e # Stop if we run out of retries
                    
                # 3. Feed the exact error back into the conversation history with prescriptive guidance
                messages.append({"role": "assistant", "content": text_out})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your previous output caused this validation error:\n{e}\n\n"
                        "Please fix the PDDL problem JSON. Remember:\n"
                        "1. Every single object or receptacle used in :init or :goal MUST be declared in 'objects'.\n"
                        "2. Use type 'obj' for manipulable items and 'location' for target receptacles (e.g. sorting_bin, pot).\n"
                        "3. Do NOT use the type word 'location' as an object name.\n"
                        "4. Keep predicate names exact (e.g. 'on-table', 'on')."
                    )
                })


class ambresFewShotPrompt(AmbresStructured):
    def __init__(self, vlm_name="qwen2.5vl:3b"):
        super().__init__(vlm_name=vlm_name)

    def reset_chat(self):
        """Use a System Prompt to teach the model, rather than fake history."""
        self.messages = []
        
        system_instruction = """You are the vision-language brain for a robot. 
            Your job is to look at the user's task and the image, extract ONLY the specific entities the robot needs to interact with, and check if the environment makes the task ambiguous.

            You must categorize all extracted entities into:
            1. "task_objects": ONLY manipulable items that the robot picks up, moves, or holds (e.g., cans, cubes, fruits, mugs).
            2. "target_locations": ONLY target receptacles, containers, or surfaces where items are placed (e.g., sorting bin, pot, bowl, drawer, table).

            CRITICAL RULES:
            1. ONLY list objects and receptacles that are explicitly part of the user's command. DO NOT list irrelevant background items.
            2. DO NOT mix locations into task_objects. Items being picked/held are 'task_objects'; receptacles being placed into are 'target_locations'.
            3. PRESERVE ADJECTIVES: You MUST keep all colors, sizes, and descriptive words exactly as the user wrote them (e.g., "red can", "sorting bin").
            4. If the task is ambiguous, your clarifying question MUST specifically mention the objects or locations causing the confusion.
            5. COMPOUND AND MULTI-STEP COMMANDS: If the command contains multiple actions or steps (e.g., "Put X in Y, then put Z in W..."):
            - Extract EVERY SINGLE manipulable object mentioned across ALL steps into "task_objects". Do NOT omit any!
            - Extract EVERY unique target receptacle mentioned across ALL steps into "target_locations". Do NOT duplicate entries in the JSON array.

            EXAMPLES OF HOW YOU MUST THINK:

            [Example 1 - Ambiguous]
            User: Task Description: Put the can in the bin.
            Please analyze the attached image and extract all relevant entities across all actions.
            Assistant: {"task_objects": ["can"], "target_locations": ["bin"]}
            User: Is the task ambiguous given the visible scene? Return a JSON object with: {"task_ambiguous": true/false, "explanation": "...", "clarifying_question": "..."}
            Assistant: {"task_ambiguous": true, "explanation": "There are multiple cans (red can, blue can) visible.", "clarifying_question": "There are multiple cans on the table. Which can would you like me to move?"}
            User: The red can.
            Assistant: {"task_objects": ["red can"], "target_locations": ["bin"]}

            [Example 2 - Clear Single-Step]
            User: Task Description: Put the green mug in the microwave.
            Please analyze the attached image and extract all relevant entities across all actions.
            Assistant: {"task_objects": ["green mug"], "target_locations": ["microwave"]}
            User: Is the task ambiguous given the visible scene? Return a JSON object with: {"task_ambiguous": true/false, "explanation": "...", "clarifying_question": "..."}
            Assistant: {"task_ambiguous": false, "explanation": "There is only one green mug and one microwave visible.", "clarifying_question": ""}

            [Example 3 - Multi-Step Chained Command]
            User: Task Description: Put the red can in the sorting bin, the yellow cube in the pot, and the blue can in the sorting bin.
            Please analyze the attached image and extract all relevant entities across all actions.
            Assistant: {"task_objects": ["red can", "yellow cube", "blue can"], "target_locations": ["sorting bin", "pot"]}
            User: Is the task ambiguous given the visible scene? Return a JSON object with: {"task_ambiguous": true/false, "explanation": "...", "clarifying_question": "..."}
            Assistant: {"task_ambiguous": false, "explanation": "The red can, yellow cube, blue can, sorting bin, and pot are all uniquely identified in the scene.", "clarifying_question": ""}

            INSTRUCTIONS: 
            Look at the NEW image and answer based ONLY on the NEW task description provided by the user below."""

        # Set this as the system prompt to anchor the model's behavior
        self.messages.append({"role": "system", "content": system_instruction})
        return
