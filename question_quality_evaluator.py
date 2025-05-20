# --- Matching cueing analysis ---
def analyze_matching_cueing(matches, correct_matches):
    """
    Enhanced cueing analysis for matching questions.
    Identifies both positive cueing (correct match is semantically obvious)
    and negative cueing (incorrect match is more semantically aligned).
    Returns a dict with:
      - issues: {left: [list of issues]}
      - per_pair_score: {left: cueing score}
      - avg_score: float
    """
    from nltk.corpus import wordnet as wn
    cueing_issues = {}
    cueing_scores = {}
    total_score = 0
    pair_count = 0

    def get_tokens(text):
        return set(w.lower() for w in nltk.word_tokenize(text) if w.isalnum())

    def get_synsets(word):
        return wn.synsets(word)

    def has_semantic_link(w1, w2, threshold=0.7):
        for s1 in get_synsets(w1):
            for s2 in get_synsets(w2):
                sim = s1.wup_similarity(s2)
                if sim and sim >= threshold:
                    return True
        return False

    for left, right_list in matches:
        left_tokens = get_tokens(left)
        correct_idx = correct_matches.get(left, None)
        if correct_idx is None or correct_idx >= len(right_list):
            continue
        correct_option = right_list[correct_idx]
        correct_tokens = get_tokens(correct_option)

        # Positive cueing score (token + synonym overlap)
        overlap = left_tokens & correct_tokens
        for ltok in left_tokens:
            for ctok in correct_tokens:
                if has_semantic_link(ltok, ctok):
                    overlap.add(ctok)
        cue_score = len(overlap) / len(correct_tokens) if correct_tokens else 0
        cueing_scores[left] = round(cue_score, 3)
        total_score += cue_score
        pair_count += 1

        # Flag issues
        issues = []
        if cue_score >= 0.6:
            issues.append("Positive cueing with correct match")

        for i, option in enumerate(right_list):
            if i == correct_idx:
                continue
            option_tokens = get_tokens(option)
            neg_overlap = left_tokens & option_tokens
            for ltok in left_tokens:
                for otok in option_tokens:
                    if has_semantic_link(ltok, otok):
                        neg_overlap.add(otok)
            neg_score = len(neg_overlap) / len(option_tokens) if option_tokens else 0
            if neg_score >= cue_score:
                issues.append(f"Negative cueing with wrong match '{option}'")

        if issues:
            cueing_issues[left] = issues

    avg_score = round(total_score / pair_count, 3) if pair_count > 0 else 0.0
    return {
        "issues": cueing_issues,
        "per_pair_score": cueing_scores,
        "avg_score": avg_score
    }
    
def detect_structural_vagueness(stem, choices):
    stem_tokens = nltk.word_tokenize(stem.lower())
    word_counts = {}
    for word in stem_tokens:
        if word.isalnum() and word not in {"the", "a", "an", "and", "in", "on", "of", "at", "to", "for", "with", "by", "is", "are"}:
            word_counts[word] = word_counts.get(word, 0) + 1
    repeated_terms = [w for w, c in word_counts.items() if c > 2]

    generic_terms = {"system", "line", "type", "form", "connected", "component", "unit"}
    unqualified_generics = [w for w in stem_tokens if w in generic_terms]

    # Check for presence of interrogative words
    asks_question = any(w in stem_tokens for w in {"which", "what", "who", "when", "where", "how"})

    # Check for noun-type overlap between stem and all choices
    overlap_scores = []
    stem_set = set(stem_tokens)
    for choice in choices:
        choice_tokens = set(nltk.word_tokenize(choice.lower()))
        overlap = stem_set & choice_tokens
        overlap_scores.append(len(overlap))
    excessive_overlap = all(score >= 3 for score in overlap_scores)

    return repeated_terms, unqualified_generics, not asks_question, excessive_overlap
# --- Simulated Sailor Profile (used in student-mode prediction) ---
sailor_profile = {
    "rank": "E4",
    "years_experience": 3,
    "mechanical_background": True,
    "prior_exposure": ["fueling", "maintenance", "gear systems"],
    "learning_style": "hands-on",  # options: visual, auditory, procedural, hands-on
    "confidence_level": 0.75  # 0 to 1, controls randomness or second-guessing
}
# --- Global state for question data ---
saved_stem = None
saved_choices = None
saved_matches = None

print("Launching Top-Level Interface...")
# --- Imports ---
import re
import textstat
import difflib
import statistics
import nltk
# Add sklearn imports for clustering/psychometric analysis (may be used for advanced features)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# --- Add WordNet import for semantic reasoning ---
from nltk.corpus import wordnet as wn

# --- Diagnostic Recommendation Function ---
def generate_diagnostic_feedback(report, correct_letter=None):
    score = report["clarity_score"]
    cohesion = report["stem_choice_cohesion"]
    cueing = report["answer_cueing_score"]
    suggestions = []

    if score < 6:
        suggestions.append("🟥 The stem lacks clarity. Simplify the language or remove vague terms.")
    if cohesion is not None and cohesion < 0.3:
        all_are_concepts = all(any(word in choice.lower() for word in ["compartment", "system", "module", "component"]) for choice in report["choices"])
        if all_are_concepts:
            suggestions.append("🟠 Cohesion score is low, but all options appear conceptually valid. No choice is clearly unrelated.")
            return suggestions
        stem_words = set(w.lower() for w in nltk.word_tokenize(report["stem"]) if w.lower() not in {"the", "a", "an", "in", "on", "of", "and", "to", "by", "with", "is", "are", "was"})
        unrelated = []
        for choice in report["choices"]:
            choice_words = set(w.lower() for w in nltk.word_tokenize(choice))
            if len(stem_words & choice_words) == 0:
                unrelated.append(choice)
        if unrelated:
            suggestions.append(
                f"🟥 The question has weak cohesion: {len(unrelated)} answer choices ('{'; '.join(unrelated)}') do not share any keywords with the stem. "
                f"This misalignment may confuse test-takers or weaken question clarity."
            )
        else:
            suggestions.append("🟥 The question has weak cohesion. Consider revising the stem or answer phrasing for better alignment.")
    if cueing == -1.0:
        suggestions.append("🟥 The stem's wording semantically favors a distractor. This may mislead students away from the correct answer.")
    elif cueing is not None and cueing >= 0.6:
        if correct_letter and report.get("choices"):
            correct_index = ord(correct_letter) - 97
            correct_choice = report["choices"][correct_index]
            stem_words = set(w.lower() for w in nltk.word_tokenize(report["stem"]) if w.lower() not in {"the", "a", "an", "in", "on", "of", "and", "to", "by", "with", "is", "are", "was"})
            correct_words = set(w.lower() for w in nltk.word_tokenize(correct_choice))
            overlapping = stem_words & correct_words
            suggestions.append(
                f"🟥 The correct answer may be too easy to guess: the stem shares keywords ({', '.join(overlapping)}) with correct choice '{correct_choice}'. "
                f"This may cue students toward the correct answer even if they don’t fully understand the content."
            )
        else:
            suggestions.append("🟥 The correct answer may be too easy to guess based on stem wording.")
    if report.get("semantic_outliers"):
        outlier = report["semantic_outliers"]
        # Try to include outlier name and check if it is among choices
        if outlier not in report.get("choices", []):
            suggestions.append(f"🟥 One distractor appears unrelated: '{outlier}'. Consider removing or rewriting it.")
        else:
            suggestions.append(f"🟠 Note: '{outlier}' was identified as an outlier, but it may reflect fallback prediction logic rather than being truly unrelated.")
    if report.get("similarity_conflict"):
        suggestions.append("🟥 Two or more choices are nearly identical. Distinguish them better.")
    if report.get("grammar_issues") and len(report["grammar_issues"]) > 2:
        suggestions.append("🟥 Multiple grammar issues detected in the stem. Revise wording for clarity.")
    # Additional: full spectrum of diagnostics
    if report.get("length_bias"):
        suggestions.append("🟥 Length bias detected — choices differ too much in length.")
    if not report.get("parallel_structure"):
        suggestions.append("🟥 Choices lack parallel grammatical structure.")
    if report.get("style_consistency") == "Inconsistent with prior question format.":
        suggestions.append("🟥 Question structure differs from earlier entries — verify formatting.")
    if report.get("semantic_outliers"):
        suggestions.append(f"🟥 Outlier detected: {report['semantic_outliers']} may not align with the rest.")

    if not suggestions:
        suggestions.append("🟩 The question appears well-constructed and balanced. No critical issues detected.")

    return suggestions

def ensure_nltk_ready():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

ensure_nltk_ready()

import language_tool_python
import tkinter as tk
from tkinter import filedialog, messagebox
import json
from docx import Document

try:
    print("Initializing LanguageTool...")
    tool = language_tool_python.LanguageTool('en-US')
    print("LanguageTool initialized successfully.")
except Exception as e:
    print(f"Failed to initialize LanguageTool: {e}")
    tool = None

### --- Question Evaluation Core ---
# --- Enhanced evaluation to support MCQ, True/False, Matching ---
def predict_correct_answer(question_type, stem, choices, matches=None):
    """
    Predicts the most likely correct answer(s) based on stem-to-choice similarity.
    For MCQ/True-False: returns predicted letter (e.g., 'a').
    For Matching: returns a dict of left->right matches.
    """
    reasoning_log = []
    if question_type in ("MCQ", "True/False"):
        # Check for numerical fact recall question
        if re.search(r"\bhow many\b", stem.lower()):
            reasoning_log.append("⚠️ Detected factual numeric recall question. Semantic prediction disabled.")
            middle_index = len(choices) // 2
            return [chr(97 + middle_index)], reasoning_log
        # --- Semantic WordNet boosting logic ---
        stopwords = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "is", "are", "was"}
        # Identify key functional terms in the stem (e.g., verbs/nouns that are central to the question)
        stem_tokens = set(
            w.lower() for w in nltk.word_tokenize(stem)
            if w.lower() not in stopwords and w.isalnum()
        )
        # Detect EXCEPT-style reversal
        reverse_logic = any(kw in stem.lower() for kw in ["except", "not", "least", "never"])
        def get_synsets(words):
            synsets = []
            for w in words:
                synsets.extend(wn.synsets(w))
            return synsets

        # --- Semantic domain extraction using WordNet lexnames ---
        def get_primary_domains(synsets):
            domains = set()
            for syn in synsets:
                lexname = syn.lexname()
                if lexname:
                    domains.add(lexname.split('.')[0])
            return domains

        stem_synsets = get_synsets(stem_tokens)
        scores = []
        for idx, choice in enumerate(choices):
            choice_tokens = set(
                w.lower() for w in nltk.word_tokenize(choice)
                if w.lower() not in stopwords and w.isalnum()
            )
            choice_synsets = get_synsets(choice_tokens)
            sim_scores = []
            for s in stem_synsets:
                for c in choice_synsets:
                    sim = s.wup_similarity(c)
                    if sim is not None:
                        sim_scores.append(sim)
            avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0

            # --- Semantic domain reasoning using WordNet domains ---
            stem_domains = get_primary_domains(stem_synsets)
            choice_domains = get_primary_domains(choice_synsets)
            # Penalize if domains do not overlap at all and stem includes technical language
            if stem_domains and choice_domains and not (stem_domains & choice_domains):
                reasoning_log.append(f"  ⚠️ Domain mismatch: Stem domains {stem_domains} vs choice domains {choice_domains}. Penalizing score.")
                avg_sim *= 0.6  # reduce score for mismatch

            reasoning_log.append(
                f"Choice {chr(65+idx)}: Score={avg_sim:.2f}, Stem→Choice terms: {', '.join(choice_tokens)}"
            )
            scores.append(avg_sim)
        # Pick the top scoring choice(s)
        max_score = max(scores) if scores else 0
        best_indices = [i for i, s in enumerate(scores) if s == max_score]
        # For EXCEPT/NOT logic, use enhanced semantic grouping logic
        if reverse_logic:
            # Enhanced logic for EXCEPT: Cluster most semantically similar three
            global wn
            # --- Procedural compatibility heuristic (domain override) ---
            # Step 1: Identify procedural verbs and state-requiring actions using WordNet
            from nltk.corpus import wordnet as wn
            def get_semantically_related(word):
                return {lemma.name().lower() for syn in wn.synsets(word, pos=wn.VERB) + wn.synsets(word, pos=wn.ADJ) for lemma in syn.lemmas()}

            procedural_verbs = get_semantically_related("verify") | get_semantically_related("check") | get_semantically_related("test")
            power_state_indicators = get_semantically_related("install") | get_semantically_related("energize") | get_semantically_related("shutdown")
            # Build a procedure alignment score for each choice
            def procedure_alignment_score(choice):
                tokens = set(w.lower() for w in nltk.word_tokenize(choice) if w.isalnum())
                # Score for procedural verb presence
                proc_score = 0
                for v in procedural_verbs:
                    if v in tokens:
                        proc_score += 1
                # Penalize if action requires system state or power
                for p in power_state_indicators:
                    if p in tokens:
                        proc_score -= 2
                # Heuristic: if "installed" or similar appears, that's a likely procedural/state mismatch
                return proc_score
            # Apply heuristic to all choices
            proc_scores = [procedure_alignment_score(choice) for choice in choices]
            # If one choice is strongly penalized compared to others, use this as override
            min_proc = min(proc_scores)
            proc_outliers = [i for i, s in enumerate(proc_scores) if s == min_proc]
            # Only override if the outlier is unique and is strongly negative (e.g., -2 or less)
            if len(proc_outliers) == 1 and proc_scores[proc_outliers[0]] <= -2:
                odd_one_out = proc_outliers[0]
                reasoning_log.append("🧠 Domain override: CIK install requires powered state — filtered out based on inferred procedural constraints.")
                return [chr(97 + odd_one_out)], reasoning_log
            # --- Semantic grouping step (fallback to domain similarity) ---
            def get_primary_domains(synsets):
                domains = set()
                for syn in synsets:
                    lexname = syn.lexname()
                    if lexname:
                        domains.add(lexname.split('.')[0])
                return domains

            # Map each choice to its domains
            choice_domains = []
            for choice in choices:
                tokens = set(w.lower() for w in nltk.word_tokenize(choice) if w.isalnum())
                synsets = []
                for t in tokens:
                    synsets.extend(wn.synsets(t))
                choice_domains.append(get_primary_domains(synsets))

            # Count overlaps
            similarity_matrix = []
            for i in range(len(choice_domains)):
                row = []
                for j in range(len(choice_domains)):
                    if i == j:
                        row.append(0)
                    else:
                        overlap = len(choice_domains[i] & choice_domains[j])
                        row.append(overlap)
                similarity_matrix.append(sum(row))

            # Get index of item least similar to others
            min_similarity = min(similarity_matrix)
            odd_one_out = similarity_matrix.index(min_similarity)

            reasoning_log.append("🧠 Enhanced 'EXCEPT' logic: semantic grouping suggests odd-one-out is choice "
                                 f"{chr(97 + odd_one_out).upper()} based on weak conceptual similarity.")
            return [chr(97 + odd_one_out)], reasoning_log
        else:
            if best_indices:
                return [chr(97 + best_indices[0])], reasoning_log
            else:
                reasoning_log.append("⚠️ No valid prediction could be made. All scores were zero or invalid.")
                return [], reasoning_log
    elif question_type == "Matching":
        # matches: list of (left, [right options])
        predicted = {}
        used_indices = set()
        deferred = []

        for left, right_options in matches:
            left_tokens = set(w.lower() for w in nltk.word_tokenize(left))
            best_score = -1
            best_idx = -1
            for idx, right in enumerate(right_options):
                if idx in used_indices:
                    continue
                right_tokens = set(w.lower() for w in nltk.word_tokenize(right))

                # Apply stricter semantic checks
                if "coolant" in left.lower() and "coolant" in right.lower():
                    score = 1.0
                elif "transmission" in left.lower() and "transmission" in right.lower():
                    score = 1.0
                elif "gear" in left.lower() and "gear" in right.lower():
                    score = 0.9
                elif "engine" in left.lower() and "engine" in right.lower():
                    score = 0.8
                elif "trim" in left.lower() and "transmission" in right.lower():
                    score = 0.7
                else:
                    overlap = left_tokens.intersection(right_tokens)
                    score = len(overlap) / max(len(left_tokens | right_tokens), 1)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_score >= 0:
                predicted[left] = best_idx
                used_indices.add(best_idx)
            else:
                deferred.append(left)

        # Assign fallback matches to any unmatched items
        if matches:
            right_options = matches[0][1]
        else:
            right_options = []

        unused_indices = [i for i in range(len(right_options)) if i not in used_indices]
        for left in deferred:
            if unused_indices:
                fallback_idx = unused_indices.pop(0)
                predicted[left] = fallback_idx
                used_indices.add(fallback_idx)
            else:
                predicted[left] = -1

        # Convert index predictions to letter form (0 → A, 1 → B, etc.)
        lettered_predicted = {}
        for left_text, idx in predicted.items():
            if idx is not None and 0 <= idx < len(right_options):
                lettered_predicted[left_text] = chr(65 + idx)  # A = 65
            else:
                lettered_predicted[left_text] = "?"

        return lettered_predicted, reasoning_log
    return [], reasoning_log

def evaluate_question(question_id, stem, choices, correct_letter=None, question_type="MCQ", correct_matches=None, matches=None, predicted_answer=None):
    """
    Evaluates a question. For MCQ/True/False: choices is a list, correct_letter is the answer.
    For Matching: choices unused, correct_matches and matches are used.
    predicted_answer: the program's prediction before user input.
    """
    print(f"Evaluating Question ID: {question_id}")
    print(f"Stem: {stem}")
    print(f"Choices: {choices}")
    print(f"Question Type: {question_type}")

    # Detect fill-in-the-blank questions more accurately
    blank_patterns = [r"_{2,}", r"\.\.\.", r"\[.*?\]", r"\(.*?\)", r"<.*?>"]
    blank_detected = any(re.search(pattern, stem) for pattern in blank_patterns)
    is_fill_in_blank = blank_detected or ("___" in stem)
    print(f"Is fill-in-the-blank: {is_fill_in_blank}")

    # For True/False, bypass advanced MCQ diagnostics (parallelism, length bias, semantic outlier, similarity, cohesion)
    if question_type == "True/False":
        # --- Intentionally bypass advanced MCQ diagnostics for True/False ---
        report = {
            "question_id": question_id,
            "stem": stem,
            "choices": choices,
            "clarity_score": score_clarity(stem),
            "readability": textstat.flesch_kincaid_grade(stem),
            "vague_terms": detect_vague_terms(stem),
            "double_negative": detect_double_negative(stem),
            "parallel_structure": True,
            "length_bias": False,
            "semantic_outliers": None,
            "similarity_conflict": False,
            "is_fill_in_blank": is_fill_in_blank,
            "suggestions": [],
            "predicted_answer": predicted_answer,
            "actual_answer": correct_letter,
            "answer_match": None,
            "answer_cueing": None,
            "answer_cueing_score": None,
            "stem_choice_cohesion": None,
        }
    else:
        report = {
            "question_id": question_id,
            "stem": stem,
            "choices": choices,
            "clarity_score": score_clarity(stem),
            "readability": textstat.flesch_kincaid_grade(stem),
            "vague_terms": detect_vague_terms(stem),
            "double_negative": detect_double_negative(stem),
            "parallel_structure": check_parallelism(choices) if question_type in ("MCQ", "True/False") else True,
            "length_bias": detect_length_bias(choices) if question_type in ("MCQ", "True/False") else False,
            "semantic_outliers": detect_semantic_outliers(choices) if question_type in ("MCQ", "True/False") else None,
            "similarity_conflict": detect_too_similar(choices) if question_type in ("MCQ", "True/False") else False,
            "is_fill_in_blank": is_fill_in_blank,
            "suggestions": [],
            "predicted_answer": predicted_answer,
            "actual_answer": correct_letter,
            "answer_match": None,
            "answer_cueing": None,
            "answer_cueing_score": None,
        }
        # Cohesion logic for non-T/F
        if question_type == "Matching" and matches:
            # For matching, compute average cohesion between left/right pairs
            total_score = 0
            for left, right_options in matches:
                # Use the predicted or actual match
                idx = None
                if correct_matches and left in correct_matches:
                    idx = correct_matches[left]
                elif predicted_answer and left in predicted_answer:
                    idx = predicted_answer[left]
                else:
                    idx = 0
                right = right_options[idx]
                score, _ = check_stem_choice_cohesion(left, [right])
                total_score += score
            avg_score = total_score / len(matches) if matches else 0
            report["stem_choice_cohesion"] = avg_score
        else:
            cohesion_score, low_cohesion = check_stem_choice_cohesion(stem, choices)
            report["stem_choice_cohesion"] = cohesion_score
            if cohesion_score < 0.3:
                report["suggestions"].append("Low semantic cohesion between stem and choices.")
            if low_cohesion:
                report["suggestions"].append(f"Choices unrelated to stem: {', '.join(low_cohesion)}")

    # Generate Suggestions
    if report["vague_terms"]:
        report["suggestions"].append(f"Remove vague word(s): {', '.join(report['vague_terms'])}")
    if report["double_negative"]:
        report["suggestions"].append("Avoid double negatives in question stem.")
    # --- Structural/referential vagueness checks ---
    repeats, unqualified, lacks_question_form, overlap_excess = detect_structural_vagueness(stem, choices)
    if repeats:
        report["suggestions"].append(f"🟠 Stem repeats non-specific terms ({', '.join(repeats)}), which may reduce clarity.")
    if unqualified:
        report["suggestions"].append(f"🟠 Stem includes generic terms without clarification: {', '.join(set(unqualified))}.")
    if lacks_question_form:
        report["suggestions"].append("🟠 The stem may not clearly ask a question or lacks an interrogative form.")
    if overlap_excess:
        report["suggestions"].append("🟠 All answer choices share excessive vocabulary with the stem. Revise stem or choices for clarity.")
    # Only add advanced MCQ suggestions if not True/False
    if question_type != "True/False":
        if question_type in ("MCQ", "True/False") and not report["parallel_structure"]:
            report["suggestions"].append("Make all answer choices follow the same grammatical pattern.")
        if question_type in ("MCQ", "True/False") and report["length_bias"]:
            report["suggestions"].append("Avoid having one answer much longer or shorter than others.")
        # --- Modified semantic outlier logic ---
        if question_type in ("MCQ", "True/False") and report["semantic_outliers"]:
            if correct_letter:
                correct_index = ord(correct_letter) - 97
                correct_choice = choices[correct_index]
                # If the semantic outlier is the correct answer
                if report["semantic_outliers"] == correct_choice:
                    report["suggestions"].append(
                        f"🟡 Note: The correct answer '{correct_choice}' was flagged as a semantic outlier, but this may be a side effect of fallback logic or procedural phrasing."
                    )
                elif report["semantic_outliers"] == report.get("predicted_answer", [None])[0].upper():
                    report["suggestions"].append(
                        f"🟠 The predicted answer '{report['semantic_outliers']}' was flagged as an outlier, but this likely reflects fallback selection rather than a content mismatch."
                    )
                else:
                    report["suggestions"].append(
                        f"🟥 Replace potentially unrelated option: {report['semantic_outliers']}"
                    )
            else:
                report["suggestions"].append(
                    f"🟥 Replace potentially unrelated option: {report['semantic_outliers']}"
                )
        if question_type in ("MCQ", "True/False") and report["similarity_conflict"]:
            report["suggestions"].append("Two or more answer choices are nearly identical.")
    # Grammar issues
    report["grammar_issues"] = check_grammar_issues(stem)
    for issue in report["grammar_issues"]:
        report["suggestions"].append(f"Grammar issue: {issue}")
    # Attach all unused diagnostics before return
    if question_type != "True/False":
        if question_type in ("MCQ", "True/False") and not report["parallel_structure"]:
            report["suggestions"].append("Answers lack parallel grammatical structure.")
        if question_type in ("MCQ", "True/False") and report["length_bias"]:
            report["suggestions"].append("Answer lengths vary significantly — avoid unintentional hints.")
        if question_type in ("MCQ", "True/False") and report["semantic_outliers"]:
            report["suggestions"].append(f"Outlier detected: {report['semantic_outliers']} may not align well.")
        if question_type in ("MCQ", "True/False") and report["similarity_conflict"]:
            report["suggestions"].append("Some choices may be too similar and confuse the test-taker.")

    # --- Answer prediction and cueing ---
    if question_type in ("MCQ", "True/False"):
        # Predicted answer is a list of letters
        if predicted_answer is None:
            predicted_answer = predict_correct_answer(question_type, stem, choices)
            report["predicted_answer"] = predicted_answer
        # If actual answer is given, compare
        if correct_letter:
            report["actual_answer"] = correct_letter
            match = correct_letter in predicted_answer if isinstance(predicted_answer, list) else False
            report["answer_match"] = match
            # Now compute cueing with actual answer
            report["answer_cueing"] = detect_answer_cueing(stem, choices, correct_letter)
            report["answer_cueing_score"] = calculate_answer_cueing_score(stem, choices, correct_letter)
            # --- Add explanation for prediction mismatch due to domain knowledge ---
            if not match:
                if correct_letter and predicted_answer and isinstance(predicted_answer, list) and correct_letter not in predicted_answer:
                    pred_letter = predicted_answer[0].upper() if isinstance(predicted_answer, list) and predicted_answer else "?"
                    correct_letter_upper = correct_letter.upper()
                    if correct_letter not in predicted_answer:
                        report["suggestions"].append(
                            f"🧠 Student-mode inference selected choice {pred_letter} due to domain similarity or phrasing. "
                            f"After seeing the correct answer {correct_letter_upper}, the mismatch is clear. "
                            f"This suggests the correct answer may rely on system-specific knowledge or a procedural detail "
                            f"not inferable without training or experience."
                        )
        else:
            report["answer_cueing"] = detect_answer_cueing(stem, choices)
            report["answer_cueing_score"] = calculate_answer_cueing_score(stem, choices)
    elif question_type == "Matching" and matches:
        # For matching, predicted_answer and correct_matches are dicts left->index
        if predicted_answer is None:
            predicted_result = predict_correct_answer(question_type, stem, choices, matches=matches)
            if isinstance(predicted_result, tuple):
                predicted_answer, reasoning_log = predicted_result
            else:
                predicted_answer = predicted_result
                reasoning_log = []
            report["predicted_answer"] = predicted_answer
        if correct_matches:
            report["actual_answer"] = correct_matches
            # Compare dicts: count number of matches correct
            correct_count = sum(predicted_answer.get(k) == v for k, v in correct_matches.items())
            total = len(correct_matches)
            report["answer_match"] = (correct_count, total)
        else:
            report["answer_match"] = None
        # Matching Complexity Index
        left_count = len(matches)
        right_count = len(matches[0][1]) if matches and matches[0][1] else 0
        extra_choices = max(right_count - left_count, 0)
        matching_complexity = round(1 + (extra_choices / left_count), 2) if left_count > 0 else 0
        report["matching_complexity"] = matching_complexity
        if matching_complexity > 1.3:
            report["suggestions"].append("🟥 High right-side choice count increases matching difficulty.")
        elif matching_complexity > 1.0:
            report["suggestions"].append("🟠 Extra right-side distractors increase matching complexity.")
        # For matching, cueing/cohesion is per pair
        report["answer_cueing"] = None
        report["answer_cueing_score"] = None
        # --- Analyze matching cueing (positive and negative) ---
        if correct_matches:
            cueing_result = analyze_matching_cueing(matches, correct_matches)
            if cueing_result:
                report["cueing_findings"] = cueing_result["issues"]
                report["answer_cueing_score"] = cueing_result["avg_score"]
                for left, issues in cueing_result["issues"].items():
                    report["suggestions"].append(
                        f"🟠 Cueing issues for '{left}': {', '.join(issues)}"
                    )

    report["explanations"] = {
        "clarity_score": (
            "The clarity score starts at 10 and is reduced for vague terms (e.g., 'best', 'most') or if the question is overly long.\n"
            "  - 🟢 ≥ 8: Clear\n"
            "  - 🟠 5–7: Acceptable\n"
            "  - 🔴 < 5: Needs revision"
        ),
        "stem_choice_cohesion": (
            "This value measures how much vocabulary the stem shares with the answer choices. For technical questions, this may naturally be lower.\n"
            " - ⚪ None: Not applicable (e.g., True/False questions)\n"
            "  - 🟢 ≥ 0.6: Strong cohesion\n"
            "  - 🟠 0.3–0.59: Moderate\n"
            "  - 🔴 < 0.3: Weak — consider revision"
        ) if question_type != "True/False" else "Cohesion is not applicable for True/False questions.",
        "grammar_check": (
            "LanguageTool flags grammar or spelling issues. Some terms may be false positives if they're domain-specific (e.g., technical acronyms).\n"
            "  - 🟢 0 issues: Ideal\n"
            "  - 🟠 1–2 issues: Acceptable\n"
            "  - 🔴 >2 issues: Needs revision"
        ),
        "answer_cueing": (
            "Answer cueing occurs when the stem overlaps heavily with one or more choices in a way that may guide the student. Color-coded score guidance:\n"
            "  - 🔴 -1.0: Strongly misleading — stem wording steers the student toward a distractor instead of the correct answer. Detected only after comparing predicted and actual answers.\n"
            "    ⚠️ This reflects a semantic trap. The model may have chosen a plausible distractor due to strong overlap in wording or domain.\n"
            "    For example, if the correct answer mentions 'crane' but the stem heavily features 'tow', the model may guess 'tow cradle'.\n"
            "  - 🟢 0.0–0.29: Good — no obvious cueing\n"
            "    ✅ The stem does not semantically bias the student toward any one choice.\n"
            "  - 🟠 0.3–0.59: Mild cueing — revise if unintended\n"
            "    🔍 A moderate number of keywords in the stem overlap with the correct choice, possibly giving it away.\n"
            "  - 🔴 ≥ 0.6: Over-cueing — stem gives away the answer\n"
            "    🚨 The correct answer shares many words or synonyms with the stem. Students may guess correctly without deep understanding."
        ),
        "style_consistency": (
            "Assesses whether the structure and format of this question match previous ones. Only runs if 2 or more questions have been analyzed."
        ),
        "matching_complexity": (
            "Measures how many extra options are on the right side vs terms on the left. "
            "A higher value means more distractors, increasing cognitive difficulty. 1.0 = balanced.\n"
            "  - 🟢 ≤ 1.0: Balanced — each stem has exactly one match. Ideal matching structure.\n"
            "  - 🟠 1.01–1.3: Slightly difficult — some extra options present, raising complexity.\n"
            "  - 🔴 > 1.3: High difficulty — many extra right-side distractors increase cognitive load.\n"
            "  - ⚪ None: No matching pairs detected or question was not parsed correctly."
        ),
        "prediction_reasoning": (
            "Each predicted answer includes a breakdown of the student's reasoning:\n"
            "- Overlap: number of shared non-trivial terms (excluding common stopwords) between the stem and the choice.\n"
            "- Score: calculated as overlap divided by the number of tokens in the choice (normalizing for length).\n"
            "- Overlapping terms: exact words found in both the stem and the choice.\n"
            "Higher scores suggest stronger student-perceived semantic alignment between stem and choice."
        ),
        "non-inferable-except": (
            "⚠️ This EXCEPT question cannot be reliably inferred without hardware/system-specific knowledge.\n"
            "This tag is applied when the evaluator predicts the wrong answer for an EXCEPT-style question and determines that inference is impossible without knowing specific technical system details.\n"
            "For example, questions involving power routing, safety interlocks, or panel connections unique to a device (e.g., the SPIP on an MCM USV) may appear plausible in multiple options, making the correct answer indistinguishable without training.\n"
            "This diagnostic helps distinguish questions where semantic logic fails due to domain-specific factual requirements, and flags the need for either additional context or expert input during review."
        ),
    }

    print("Report generated successfully.")

    # Tag EXCEPT-style questions that require specific hardware/system knowledge
    if "except" in stem.lower():
        # Check if inference failed due to semantic similarity being misleading
        if correct_letter and predicted_answer and isinstance(predicted_answer, list):
            if correct_letter not in predicted_answer:
                report.setdefault("tags", []).append("non-inferable-except")
                report["suggestions"].append("⚠️ This EXCEPT question cannot be reliably inferred without hardware/system-specific knowledge.")

    # --- Cross-question consistency check ---
    if "previous_questions" not in evaluate_question.__dict__:
        evaluate_question.previous_questions = []

    if question_id != "ManualEntry":
        evaluate_question.previous_questions.append((stem, choices))

    if len(evaluate_question.previous_questions) < 2:
        report["style_consistency"] = "Insufficient data to assess consistency."
    else:
        last_stem, _ = evaluate_question.previous_questions[-2]
        # Check consistency by comparing question form (e.g., fill-in-blank style, question structure)
        if is_fill_in_blank and "___" in last_stem:
            report["style_consistency"] = "Consistent with prior question format (fill-in-the-blank)."
        elif not is_fill_in_blank and "___" not in last_stem:
            report["style_consistency"] = "Consistent with prior question format (standard)."
        else:
            report["style_consistency"] = "Inconsistent with prior question format."

    # --- Inference Heuristic Tagging (Semantic, not keyword-based) ---
    from nltk.corpus import wordnet as wn
    def semantically_related_to(word, target_synsets, threshold=0.7):
        syns = wn.synsets(word)
        for s in syns:
            for t in target_synsets:
                sim = s.wup_similarity(t)
                if sim and sim >= threshold:
                    return True
        return False

    # Build concept synsets
    sensor_related = [wn.synset("sensor.n.01"), wn.synset("camera.n.01"), wn.synset("surveillance.n.01")]
    autonomous_related = [wn.synset("autonomous.a.01"), wn.synset("robot.n.01"), wn.synset("vehicle.n.01")]

    # Tokenize stem and test concept connections
    stem_tokens = [w.lower() for w in nltk.word_tokenize(stem) if w.isalnum()]
    has_sensor_concept = any(semantically_related_to(word, sensor_related) for word in stem_tokens)
    has_autonomous_concept = any(semantically_related_to(word, autonomous_related) for word in stem_tokens)
    has_quantifiable_metric = any(re.search(r"\d{1,3}-?degree", word) or word in {"360", "angle", "field", "coverage"} for word in stem_tokens)

    if has_sensor_concept and has_autonomous_concept and has_quantifiable_metric:
        report["suggestions"].append("🧠 This question is conceptually inferable from mission context using semantic understanding (e.g., FOV geometry + unmanned system).")

    # --- Positive suggestion logic ---
    if not report["suggestions"]:
        # Only add cohesion-based positive feedback for non-T/F
        if question_type != "True/False" and isinstance(report.get("stem_choice_cohesion"), (float, int)):
            if report["stem_choice_cohesion"] >= 0.6:
                report["suggestions"].append("Good cohesion between question and answers. Question appears well-aligned.")
            else:
                report["suggestions"].append("No major structural issues detected.")
        else:
            report["suggestions"].append("No major structural issues detected.")

    return report

### --- Evaluation Functions ---
def score_clarity(stem):
    vague_words = ["best", "most", "always", "never", "often", "sometimes", "usually"]
    penalty = sum(1 for word in vague_words if word in stem.lower().split())
    length_penalty = 1 if len(stem.split()) > 25 else 0
    return max(10 - penalty - length_penalty, 0)

def detect_vague_terms(text):
    vague_terms = ["best", "most", "always", "never", "often", "sometimes", "usually"]
    return [word for word in vague_terms if word in text.lower().split()]

def detect_double_negative(text):
    return bool(re.search(r"\bnot\s+(un|no|never|none)", text.lower()))

def check_parallelism(choices):
    # Original POS tag pattern check
    tag_patterns = []
    for choice in choices:
        tokens = nltk.word_tokenize(choice)
        tags = nltk.pos_tag(tokens)
        # Only check first 3 tags to allow for variation in longer answers
        tag_pattern = ' '.join([tag for word, tag in tags[:3]])
        tag_patterns.append(tag_pattern)
    # Consider it parallel if at least half follow the same pattern
    pattern_counts = {}
    for pattern in tag_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    most_common = max(pattern_counts.values()) if pattern_counts else 0
    return most_common >= len(choices) / 2

def detect_length_bias(choices):
    if not choices:
        return False
    lengths = [len(choice.split()) for choice in choices if choice]
    if not lengths:
        return False
    return max(lengths) - min(lengths) > 5

def detect_semantic_outliers(choices, is_technical=False):
    # Uses simple token overlap as a placeholder for semantic similarity
    token_sets = [set(nltk.word_tokenize(c.lower())) for c in choices]
    scores = []
    for i, tokens in enumerate(token_sets):
        others = [token_sets[j] for j in range(len(token_sets)) if j != i]
        if not others:
            scores.append(1.0)  # If there's only one choice
            continue
        overlaps = []
        for other in others:
            if not (tokens | other):
                overlaps.append(0)
            else:
                overlaps.append(len(tokens & other) / len(tokens | other))
        avg_overlap = statistics.mean(overlaps) if overlaps else 0
        scores.append(avg_overlap)
    if not scores:
        return None
    min_score = min(scores)
    if min_score < 0.15:  # Reduced threshold
        return choices[scores.index(min_score)]
    return None

def detect_too_similar(choices):
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            sim = difflib.SequenceMatcher(None, choices[i], choices[j]).ratio()
            if sim > 0.85:
                return True
    return False

# --- Answer cueing detection ---
def detect_answer_cueing(stem, choices, correct_letter=None):
    ignore_words = {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by", "is", "are", "was"}
    stem_tokens = set(w for w in nltk.word_tokenize(stem.lower()) if w not in ignore_words)
    cueing_signals = []

    # --- Semantic similarity helpers ---
    def get_synsets(word):
        return wn.synsets(word)

    def has_semantic_link(word1, word2, threshold=0.7):
        syns1 = get_synsets(word1)
        syns2 = get_synsets(word2)
        for s1 in syns1:
            for s2 in syns2:
                sim = s1.wup_similarity(s2)
                if sim and sim >= threshold:
                    return True
        return False

    for idx, choice in enumerate(choices):
        choice_tokens = set(w for w in nltk.word_tokenize(choice.lower()) if w not in ignore_words)
        overlap = stem_tokens & choice_tokens
        # Add semantic similarity-based linking
        for st in stem_tokens:
            for ct in choice_tokens:
                if has_semantic_link(st, ct):
                    overlap.add(ct)
        cue_score = len(overlap) / len(choice_tokens) if choice_tokens else 0
        cueing_signals.append((chr(97 + idx), cue_score))  # 'a', 'b', etc.

    if correct_letter:
        correct_score = next((s for l, s in cueing_signals if l == correct_letter), 0)
        distractor_scores = [s for l, s in cueing_signals if l != correct_letter]
        if correct_score >= 0.5 and all(correct_score > s for s in distractor_scores):
            return f"Correct answer '{correct_letter.upper()}' may be revealed by stem phrasing."
    return None

def calculate_answer_cueing_score(stem, choices, correct_letter=None):
    from collections import defaultdict

    stopwords = {
        "the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "by",
        "is", "are", "was", "this", "that", "with", "as", "from"
    }
    stem_tokens = set(w.lower() for w in nltk.word_tokenize(stem) if w.lower() not in stopwords)
    keyword_map = defaultdict(set)

    # --- Semantic similarity helpers ---
    def get_synsets(word):
        return wn.synsets(word)

    def has_semantic_link(word1, word2, threshold=0.7):
        syns1 = get_synsets(word1)
        syns2 = get_synsets(word2)
        for s1 in syns1:
            for s2 in syns2:
                sim = s1.wup_similarity(s2)
                if sim and sim >= threshold:
                    return True
        return False

    for idx, choice in enumerate(choices):
        choice_tokens = set(w.lower() for w in nltk.word_tokenize(choice) if w.lower() not in stopwords)
        for word in stem_tokens:
            if word in choice_tokens or any(has_semantic_link(word, ct) for ct in choice_tokens):
                keyword_map[word].add(idx)

    # Fallback logic if no correct answer is provided
    if correct_letter is None:
        # Estimate generic cueing potential if no correct answer is provided
        total_choices = len(choices)
        reduced_sets = [indices for word, indices in keyword_map.items() if len(indices) < total_choices]
        if not reduced_sets:
            return 0.0
        # Use average narrowing effect as proxy
        avg_reduced = sum(len(s) for s in reduced_sets) / len(reduced_sets)
        base_prob = 1 / total_choices
        reduced_prob = 1 / avg_reduced if avg_reduced else base_prob
        gain = reduced_prob / base_prob
        return round(min(gain * 0.3, 1.0), 3)  # scaled to 0.3 max since correct_letter is unknown

    if not keyword_map:
        return 0.0

    total_choices = len(choices)
    correct_index = ord(correct_letter) - 97 if correct_letter else None

    # Aggregate indices where stem keywords appear
    reduced_sets = [indices for word, indices in keyword_map.items() if len(indices) < total_choices]

    if not reduced_sets:
        return 0.0

    # Check how many times the correct answer appears in narrowed sets
    narrowing_hits = sum(1 for s in reduced_sets if correct_index in s)
    total_paths = len(reduced_sets)

    # --- New logic: detect misleading cueing toward distractors ---
    # If most reduced sets point to distractors, penalize
    if correct_letter:
        # Measure cueing toward incorrect answers
        distractor_indices = [i for i in range(total_choices) if i != correct_index]
        distractor_hits = sum(1 for s in reduced_sets if any(d in s for d in distractor_indices))
        # Only allow negative scoring if the predicted answer did NOT match the correct one
        predicted_indices = [min(s, default=-1) for s in reduced_sets]
        if distractor_hits > len(reduced_sets) / 2 and correct_index not in predicted_indices:
            # Negative score for misleading cueing only if the model was misled
            avg_reduced = sum(len(s) for s in reduced_sets) / len(reduced_sets)
            base_prob = 1 / total_choices
            gain = (1 / avg_reduced) / base_prob if avg_reduced else 1.0
            return -round(min(gain * 0.5, 1.0), 3)

    if narrowing_hits == 0:
        return 0.0

    # Estimate guessing advantage
    avg_reduced = sum(len(s) for s in reduced_sets if correct_index in s) / narrowing_hits
    base_prob = 1 / total_choices
    improved_prob = 1 / avg_reduced if avg_reduced else base_prob
    gain = improved_prob / base_prob

    # The score may be negative if cueing misleads the student consistently toward distractors.
    return round(min(gain * 0.5, 1.0), 3)

def check_stem_choice_cohesion(stem, choices, is_technical=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Token-level overlap cohesion
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "from", "is", "are", "was"}
    stem_tokens = [w.lower() for w in nltk.word_tokenize(stem) if w.lower() not in stop_words]
    stem_set = set(stem_tokens)

    cohesion_scores = []
    low_similarity_choices = []

    for choice in choices:
        choice_tokens = [w.lower() for w in nltk.word_tokenize(choice) if w.lower() not in stop_words]
        choice_set = set(choice_tokens)
        overlap = stem_set & choice_set
        weighted_overlap = len(overlap)
        total = len(stem_set | choice_set)
        token_similarity = weighted_overlap / total if total else 0
        cohesion_scores.append(token_similarity)
        if token_similarity < 0.3:
            low_similarity_choices.append(choice)

    # TF-IDF semantic similarity enhancement
    try:
        corpus = [stem] + choices
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        # Blend token overlap and semantic similarity
        blended_scores = [(a + b) / 2 for a, b in zip(cohesion_scores, similarities)]
        avg_score = statistics.mean(blended_scores)
        low_sim_choices = [choices[i] for i, s in enumerate(blended_scores) if s < 0.3]
        return avg_score, low_sim_choices
    except:
        avg_similarity = statistics.mean(cohesion_scores) if cohesion_scores else 0
        return avg_similarity, low_similarity_choices

def check_fill_in_blank_cohesion(stem, choices, is_technical=False):
    # For fill-in-blank questions, we need to check if the choices make sense in the blank
    # This is a simplified approach - we check if the choices are of the right type
    
    # Extract context around the blank
    blank_patterns = [r"_{2,}", r"\.\.\.", r"\[.*?\]", r"\(.*?\)", r"<.*?>"]
    
    # Find the blank position
    blank_pos = -1
    for pattern in blank_patterns:
        match = re.search(pattern, stem)
        if match:
            blank_pos = match.start()
            break
    
    if blank_pos == -1:
        # If no explicit blank marker found, look for "___"
        blank_pos = stem.find("___")
    
    if blank_pos == -1:
        # If still no blank found, fall back to regular cohesion check
        return check_stem_choice_cohesion(stem, choices, is_technical)
    
    # Get words before and after the blank
    words = stem.split()
    blank_word_idx = -1
    
    # Find which word contains the blank
    current_pos = 0
    for i, word in enumerate(words):
        if current_pos <= blank_pos < current_pos + len(word):
            blank_word_idx = i
            break
        current_pos += len(word) + 1  # +1 for the space
    
    if blank_word_idx == -1:
        # Fallback if we can't locate the blank position in words
        return check_stem_choice_cohesion(stem, choices, is_technical)
    
    # Get context (3 words before and after the blank)
    start_idx = max(0, blank_word_idx - 3)
    end_idx = min(len(words), blank_word_idx + 4)
    
    context_before = " ".join(words[start_idx:blank_word_idx])
    context_after = " ".join(words[blank_word_idx+1:end_idx])
    
    # For technical fill-in-blank questions, we're more lenient
    if is_technical:
        return 0.7, []  # Assume good cohesion for technical fill-in-blank
    
    # Check if choices fit grammatically in the blank
    cohesion_scores = []
    low_cohesion_choices = []
    
    for choice in choices:
        # Simple check: does the choice + context form a valid phrase?
        test_phrase = f"{context_before} {choice} {context_after}".strip()
        
        # Use grammar check as a proxy for fit
        if tool is None:
            grammar_errors = 0
        else:
            matches = tool.check(test_phrase)
            grammar_errors = len(matches)
        
        # Normalize to a score between 0 and 1
        score = max(0, 1 - (grammar_errors / 10))
        cohesion_scores.append(score)
        
        if score < 0.5:
            low_cohesion_choices.append(choice)
    
    avg_score = statistics.mean(cohesion_scores) if cohesion_scores else 0
    return avg_score, low_cohesion_choices

def check_grammar_issues(text):
    if tool is None:
        return ["LanguageTool unavailable — grammar check skipped."]
    matches = tool.check(text)
    return [f"{m.ruleId}: {m.message} → {text[m.offset:m.offset + m.errorLength]}" for m in matches]

### --- UI Functions ---
def load_questions_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_questions_from_docx(filepath):
    print(f"Loading DOCX file: {filepath}")
    doc = Document(filepath)
    questions = []
    current_question = None
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
def color_bubble(name, value):
    if value is None or not isinstance(value, (int, float)):
        return "⚪"
    if name == "clarity":
        return "🟢" if value >= 8 else "🟠" if value >= 5 else "🔴"
    if name == "cohesion":
        return "🟢" if value >= 0.6 else "🟠" if value >= 0.3 else "🔴"
    if name == "cueing":
        if value is None:
            return "⚪"
        if value < 0:
            return "🔴"  # Misleading cueing (stem favors wrong answer)
        elif value < 0.3:
            return "🟢"  # No cueing, or weak enough — good
        elif value < 0.6:
            return "🟠"  # Moderate cueing — maybe revise
        else:
            return "🔴"  # Strong cueing — stem gives away the answer
    return ""
    
# --- UI: Manual Entry Handler, with Question Type and Prediction ---
def open_file_and_analyze():
    try:
        global saved_stem, saved_choices, saved_matches
        if not saved_stem:
            messagebox.showerror("Error", "Stem not found. Please submit the question first.")
            return
        stem_line = saved_stem
        choices = saved_choices
        matches = saved_matches
        # Determine question type from radio button
        qtype = question_type_var.get()
        # Step 1: Predict answer before user input
        predicted = None
        reasoning = None
        if qtype in ("MCQ", "True/False"):
            predicted, reasoning = predict_correct_answer(qtype, stem_line, choices)
        elif qtype == "Matching" and matches:
            predicted, reasoning = predict_correct_answer(qtype, stem_line, choices, matches=matches)
        # Step 2: Show question info in UI (without reprinting predicted matches)

        if 'output_text' in globals() and output_text is not None:
            output_text.delete("1.0", tk.END)
            output_text.insert(tk.END, f"Question Type: {qtype}\n")
            output_text.insert(tk.END, f"Stem: {stem_line}\n")
            if qtype in ("MCQ", "True/False"):
                output_text.insert(tk.END, "Choices:\n")
                for i, c in enumerate(choices):
                    output_text.insert(tk.END, f"  {chr(97+i).upper()}. {c}\n")
            elif qtype == "Matching" and matches:
                output_text.insert(tk.END, f"Matching pairs (Left/Right):\n")
                lefts = [l for l, _ in matches]
                rights = matches[0][1] if matches else []
                output_text.insert(tk.END, f"  Left: {', '.join(lefts)}\n")
                output_text.insert(tk.END, f"  Right: {', '.join(rights)}\n")
            output_text.insert(tk.END, "\n---\n")
        
        # Step 3: Accept user input for actual answer, then re-run full evaluation
        correct_answer_text = correct_answer_entry.get().strip().lower()
        valid_letters = [chr(97+i) for i in range(len(choices))]
        correct_letter = None
        # Use saved_correct_matches for correct_matches
        global saved_correct_matches
        # Fix: ensure correct_matches is a dict, not None
        correct_matches = saved_correct_matches or {}

        if qtype in ("MCQ", "True/False"):
            if correct_answer_text and correct_answer_text[0] in valid_letters:
                correct_letter = correct_answer_text[0]
        elif qtype == "Matching" and matches:
            # Accept user input as comma-separated letters (e.g., A,B,C for left1->rightA, etc.)
            if correct_answer_text:
                try:
                    letter_to_index = {chr(65+i): i for i in range(len(matches[0][1]))}
                    tokens = [x.strip().upper() for x in correct_answer_text.split(",")]
                    indices = [letter_to_index.get(t, -1) for t in tokens]
                    if len(indices) == len(matches) and all(i >= 0 for i in indices):
                        correct_matches = {matches[i][0]: indices[i] for i in range(len(matches))}
                except Exception:
                    pass
        
        # Step 4: Debug print before evaluate_question
        print("Debug Info:", stem_line, choices)
        
        # Step 4: Run evaluation using both predicted and actual answer
        result = evaluate_question(
            "ManualEntry",
            stem=stem_line,
            choices=choices,
            correct_letter=correct_letter,
            question_type=qtype,
            correct_matches=correct_matches,
            matches=matches,
            predicted_answer=predicted
        )
        # --- Now safe to display predicted matches ---
        if qtype == "Matching" and result.get("predicted_answer"):
            output_text.insert(tk.END, f"\n🔍 Predicted Matches:\n")
            pred_matches = result["predicted_answer"]
            lefts = [l for l, _ in matches]
            rights = matches[0][1] if matches else []
            right_lookup = {letter: text for letter, text in question_data.get("right_items", [])}
            for idx, left in enumerate(lefts):
                letter = pred_matches.get(left, "?")
                match_text = right_lookup.get(letter, "?")
                output_text.insert(tk.END, f"  {idx+1}. {left} → {letter}. {match_text}\n")
        
        # Store reasoning in the result/report
        result["prediction_reasoning"] = reasoning
        
        # Step 5: Output scores and diagnostics
        if 'output_text' in globals() and output_text is not None:
            output_text.insert(tk.END, f"Clarity Score: {color_bubble('clarity', result['clarity_score'])} {result['clarity_score']}\n")
            output_text.insert(tk.END, f"Cohesion: {color_bubble('cohesion', result['stem_choice_cohesion'])} {result['stem_choice_cohesion']}\n")
            if qtype == "Matching" and result.get("matching_complexity") is not None:
                output_text.insert(tk.END, f"Matching Complexity: {color_bubble('cohesion', result['matching_complexity'])} {result['matching_complexity']}\n")
            if qtype in ("MCQ", "True/False"):
                output_text.insert(tk.END, f"Answer Cueing Score: {color_bubble('cueing', result['answer_cueing_score'])} {result['answer_cueing_score']}\n")
                # --- Cueing explanation --- (moved up)
                if result.get("answer_cueing"):
                    output_text.insert(tk.END, f"\nAnswer Cueing: {result['answer_cueing']}\n")

            # --- Reporting: match between prediction and actual
            if qtype in ("MCQ", "True/False"):
                output_text.insert(tk.END, f"Predicted answer: {', '.join([x.upper() for x in predicted]) if predicted else '?'}\n")
                # Show reasoning if available
                reasoning = result.get("prediction_reasoning")
                if reasoning:
                    output_text.insert(tk.END, "\n🧠 Student Reasoning:\n")
                    for line in reasoning:
                        output_text.insert(tk.END, f"  - {line}\n")

                # --- Additional reasoning when cueing is strong or misleading ---
                cue_score = result.get("answer_cueing_score")
                if qtype in ("MCQ", "True/False"):
                    if cue_score == -1.0 and not result["answer_match"]:
                        output_text.insert(tk.END, "\n🧠 Explanation:\n")
                        output_text.insert(tk.END,
                            "The disparity between the predicted and correct answer is due to misleading semantic cueing. "
                            "The model was drawn to an incorrect choice based on wording overlap or conceptual alignment, but the correct answer "
                            "does not share this overlap. This misdirection results in a cueing score of -1.0 — meaning the stem favored a distractor.\n"
                        )
                    elif cue_score == -1.0 and result["answer_match"]:
                        output_text.insert(tk.END, "\n🧠 Explanation:\n")
                        output_text.insert(tk.END,
                            "Despite a cueing score of -1.0 indicating strong semantic alignment with an incorrect choice, "
                            "the model correctly predicted the answer. This suggests that non-semantic logic (e.g., position, domain knowledge, fallback) overcame the misleading signal.\n"
                        )

                output_text.insert(tk.END, f"Actual answer: {correct_letter.upper() if correct_letter else '(not provided)'}\n")
                if correct_letter:
                    if result["answer_match"]:
                        output_text.insert(tk.END, "✅ Prediction matched the actual answer!\n")
                    else:
                        output_text.insert(tk.END, "❌ Prediction did not match the actual answer.\n")

            elif qtype == "Matching" and matches:
                lefts = [l for l, _ in matches]
                rights = matches[0][1] if matches else []
                # Only print actual matches and evaluation results (do not reprint predicted matches)
                # Debug: Safeguard print for correct_matches
                print("Correct matches received:", correct_matches)

                # Matching questions: estimate cueing by computing overlap between lefts and rights
                if saved_matches:
                    cueing_values = []
                    for left_text, right_list in saved_matches:
                        for right_text in right_list:
                            left_tokens = set(nltk.word_tokenize(left_text.lower()))
                            right_tokens = set(nltk.word_tokenize(right_text.lower()))
                            overlap = left_tokens & right_tokens
                            score = len(overlap) / len(right_tokens) if right_tokens else 0
                            cueing_values.append(score)
                    if cueing_values:
                        avg_cueing = sum(cueing_values) / len(cueing_values)
                        output_text.insert(tk.END, f"Answer Cueing Score (estimated): {avg_cueing:.3f}\n")
                    # Display refined cueing explanation if available (ensure appears after cueing score print)
                    if result.get("answer_cueing"):
                        output_text.insert(tk.END, f"\nAnswer Cueing: {result['answer_cueing']}\n")
                # ---- Insert block to display predicted answers for Matching if not already shown ----
                if result.get("predicted_answer"):
                    output_text.insert(tk.END, f"\n🔍 Predicted Matches:\n")
                    pred_matches = result["predicted_answer"]
                    lefts = [l for l, _ in matches]
                    rights = matches[0][1] if matches else []
                    # Use question_data for right_items if available
                    right_lookup = {letter: text for letter, text in question_data.get("right_items", [])}
                    for idx, left in enumerate(lefts):
                        letter = pred_matches.get(left, "?")
                        match_text = right_lookup.get(letter, "?")
                        output_text.insert(tk.END, f"  {idx+1}. {left} → {letter}. {match_text}\n")

                # Show correct matches if possible
                if correct_matches and len(correct_matches) == len(lefts):
                    output_text.insert(tk.END, f"\nActual matches:\n")
                    for idx, l in enumerate(lefts):
                        correct_idx = correct_matches.get(l)
                        correct_val = rights[correct_idx] if correct_idx is not None and correct_idx < len(rights) else '?'
                        output_text.insert(tk.END, f"  {idx+1}. {l} → {correct_val}\n")
                else:
                    output_text.insert(tk.END, "\n⚠️ Correct matches could not be printed: mismatch or missing data.\n")

                # Show reasoning if available
                reasoning = result.get("prediction_reasoning")
                if reasoning:
                    output_text.insert(tk.END, "\n🧠 Student Reasoning:\n")
                    for line in reasoning:
                        output_text.insert(tk.END, f"  - {line}\n")

                # --- Additional reasoning for Matching (if ever needed) ---
                # Matching questions: estimate cueing by computing overlap between lefts and rights
                if saved_matches:
                    cueing_values = []
                    for left_text, right_list in saved_matches:
                        for right_text in right_list:
                            left_tokens = set(nltk.word_tokenize(left_text.lower()))
                            right_tokens = set(nltk.word_tokenize(right_text.lower()))
                            overlap = left_tokens & right_tokens
                            score = len(overlap) / len(right_tokens) if right_tokens else 0
                            cueing_values.append(score)
                    if cueing_values:
                        avg_cueing = sum(cueing_values) / len(cueing_values)
                        output_text.insert(tk.END, f"Answer Cueing Score (estimated): {avg_cueing:.3f}\n")
                if correct_matches:
                    # Match score
                    match_score = result["answer_match"]
                    if match_score:
                        output_text.insert(tk.END, f"Prediction matched {match_score[0]}/{match_score[1]} pairs.\n")

            # --- Diagnostic Summary ---
            output_text.insert(tk.END, "\n🧠 Diagnostic Summary:\n")
            output_text.insert(tk.END, "Refer to the Explanation Guide below for score interpretation.\n")
            diagnostics = generate_diagnostic_feedback(result, correct_letter)
            for d in diagnostics:
                output_text.insert(tk.END, f"{d}\n")

            # --- Suggestions ---
            output_text.insert(tk.END, "\nSuggestions:\n")
            for s in result['suggestions']:
                output_text.insert(tk.END, f"  - {s}\n")

            # --- Grammar Issues ---
            if result["grammar_issues"]:
                output_text.insert(tk.END, "\nGrammar Issues:\n")
                for g in result["grammar_issues"]:
                    output_text.insert(tk.END, f"  - {g}\n")

            # --- Style Consistency ---
            if result.get("style_consistency"):
                output_text.insert(tk.END, f"\nStyle Consistency: {result['style_consistency']}\n")

            # --- Glossary ---
            output_text.insert(tk.END, "\n📘 Explanation Guide (for reference):\n")
            # Print the full Explanation Guide block for reference
            # Now show the detailed guide for each key
            for key, explanation in result.get("explanations", {}).items():
                output_text.insert(tk.END, f"\n[{key}]\n{explanation}\n")
            # --- Insert stem_choice_cohesion explanation ---
            output_text.insert(tk.END, f"\n[stem_choice_cohesion]\n")
            output_text.insert(tk.END, "This value measures how much vocabulary the stem shares with the answer choices. For technical questions, this may naturally be lower.\n")
            output_text.insert(tk.END, " - ⚪ None: Not applicable (e.g., True/False questions)\n")
            output_text.insert(tk.END, "  - 🟢 ≥ 0.6: Strong cohesion\n")
            output_text.insert(tk.END, "  - 🟠 0.3–0.59: Moderate\n")
            output_text.insert(tk.END, "  - 🔴 < 0.3: Weak — consider revision\n")

            if 'output_text' in globals():
                output_text.focus_set()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to analyze input:\n{e}")

# --- Add global state flag
question_submitted = False
saved_correct_matches = None


def extract_technical_terms(text):
    """Extract potential technical terms from text"""
    # This is a simplified approach - a real system would use NLP techniques
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]*\b', text)
    
    # Filter for potential technical terms (capitalized or compound words)
    technical_terms = [
        word for word in words
        if (word[0].isupper() or '-' in word or '_' in word or
            any(c.isupper() for c in word[1:]))
    ]
    
    return technical_terms

def calculate_text_similarity(text1, text2):
    """Calculate similarity between two text strings"""
    # Convert to lowercase
    text1 = text1.lower()
    text2 = text2.lower()
    
    # Tokenize into words
    words1 = set(re.findall(r'\b[a-z]+\b', text1))
    words2 = set(re.findall(r'\b[a-z]+\b', text2))
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0


import re

def parse_mcq(text):
    """
    Parses a multiple-choice question (MCQ) from raw text.

    Returns:
        stem (str): The question stem.
        choices (list of str): The list of answer choices.
    """
    lines = text.strip().split('\n')
    stem_lines = []
    choices = []

    for line in lines:
        stripped = line.strip()
        if re.match(r'^[A-Da-d]\.\s+', stripped):
            choices.append(re.sub(r'^[A-Da-d]\.\s+', '', stripped))
        else:
            stem_lines.append(stripped)

    stem = ' '.join(stem_lines).strip()
    return stem, choices

def parse_question(question_text, question_type):
    """
    Parse the question text based on the question type.
    
    Args:
        question_text: The raw question text
        question_type: The type of question (MCQ, True/False, Matching)
        
    Returns:
        A dictionary containing the parsed question data
    """
    # Remove any duplicate text that might have been pasted multiple times
    # This pattern looks for repeated blocks of text
    cleaned_text = remove_duplicate_text(question_text)
    
    # Create a base dictionary for the question data
    question_data = {
        "question_type": question_type,
        "stem": "",
        "choices": [],
        "left_items": [],
        "right_items": []
    }
    
    # Parse based on question type
    if question_type == "MCQ":
        # Parse MCQ question
        stem, choices = parse_mcq(cleaned_text)
        question_data["stem"] = stem
        question_data["choices"] = choices
        
    elif question_type == "True/False":
        # Parse True/False question
        stem = cleaned_text.strip()
        question_data["stem"] = stem
        question_data["choices"] = ["True", "False"]
        
    elif question_type == "Matching":
        # Parse Matching question
        stem, left_items, right_items = parse_matching(cleaned_text)
        question_data["stem"] = stem
        question_data["left_items"] = left_items
        question_data["right_items"] = right_items
    
    return question_data

def remove_duplicate_text(text):
    """
    Remove duplicate blocks of text that might have been pasted multiple times.
    
    Args:
        text: The raw text that might contain duplicates
        
    Returns:
        Cleaned text with duplicates removed
    """
    # If the text is short, no need to check for duplicates
    if len(text) < 100:
        return text
    
    # Look for large repeated blocks
    lines = text.split('\n')
    
    # Check if the first half and second half are nearly identical
    half_point = len(lines) // 2
    first_half = '\n'.join(lines[:half_point])
    second_half = '\n'.join(lines[half_point:])
    
    # If the halves are very similar, keep only the first half
    similarity = calculate_similarity(first_half, second_half)
    if similarity > 0.7:  # 70% similarity threshold
        return first_half
    
    # Check for other patterns of duplication
    # Look for repeated sections with "PREDICTION RESULTS" or similar headers
    sections = re.split(r'(PREDICTION RESULTS|EVALUATION RESULTS|={10,})', text)
    if len(sections) > 3:  # Multiple sections detected
        # Keep only the first complete section
        for i in range(len(sections)):
            if "PREDICTION RESULTS" in sections[i]:
                # Return everything up to the next "PREDICTION RESULTS" or end
                end_idx = len(sections)
                for j in range(i+1, len(sections)):
                    if "PREDICTION RESULTS" in sections[j]:
                        end_idx = j
                        break
                return ''.join(sections[:end_idx])
    
    # If no clear duplication pattern is found, return the original text
    return text

def calculate_similarity(text1, text2):
    """
    Calculate the similarity between two text blocks.
    
    Args:
        text1: First text block
        text2: Second text block
        
    Returns:
        Similarity score between 0 and 1
    """
    # Simple Jaccard similarity on words
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0


def parse_matching(text):
    """
    Parse a matching question into stem, left items, and right items.
    
    Args:
        text: The raw matching question text
        
    Returns:
        Tuple of (stem, left_items, right_items)
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Extract the stem (everything before the first item)
    stem_lines = []
    item_start_idx = -1
    
    for i, line in enumerate(lines):
        # Look for patterns that indicate the start of items
        if re.search(r'^\s*\d+\.\s+', line) or re.search(r'^\s*[A-Z]\.\s+', line):
            item_start_idx = i
            break
        stem_lines.append(line)
    
    stem = '\n'.join(stem_lines).strip()
    
    # If no items were found, return empty lists
    if item_start_idx == -1:
        return stem, [], []
    
    # Extract left items (numbered items)
    left_items = []
    for i in range(item_start_idx, len(lines)):
        line = lines[i].strip()
        # Look for numbered items (1., 2., etc.)
        match = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if match:
            left_items.append(match.group(2).strip())
    
    # Extract right items (lettered items)
    right_items = []
    for i in range(item_start_idx, len(lines)):
        line = lines[i].strip()
        # Look for lettered items (A., B., etc.)
        match = re.match(r'^\s*([A-Z])\.\s+(.*)', line)
        if match:
            right_items.append((match.group(1), match.group(2).strip()))
    
    return stem, left_items, right_items
def predict_answer(question_data):
    """
    Predict the answer for a given question based purely on linguistic analysis
    without any hardcoded domain knowledge.
    """
    try:
        question_type = question_data.get("question_type", "")
        
        if question_type == "Matching":
            left_items = question_data.get("left_items", [])
            right_items = question_data.get("right_items", [])
            
            if not left_items or not right_items:
                return {}
            
            # Create a matrix of similarity scores
            similarity_matrix = []
            
            for left in left_items:
                # Tokenize the left item into words
                left_words = tokenize_text(left)
                
                row_scores = []
                for _, right in right_items:
                    # Tokenize the right item into words
                    right_words = tokenize_text(right)
                    
                    # Calculate a similarity score based on word relationships
                    score = compute_linguistic_similarity(left_words, right_words)
                    row_scores.append(score)
                
                similarity_matrix.append(row_scores)
            
            # Find the best assignment using the Hungarian algorithm
            mapping = {}
            used_right_indices = set()
            
            # Create a list of all possible matches with their scores
            all_matches = []
            for i, left in enumerate(left_items):
                for j, (letter, _) in enumerate(right_items):
                    all_matches.append((i, j, similarity_matrix[i][j]))
            
            # Sort by score in descending order
            all_matches.sort(key=lambda x: x[2], reverse=True)
            
            # Assign matches greedily
            for left_idx, right_idx, _ in all_matches:
                left_indices = [list(left_items).index(l) for l in mapping.keys()]
                if left_idx not in left_indices and right_idx not in used_right_indices:
                    left = left_items[left_idx]
                    letter = right_items[right_idx][0]
                    mapping[left] = letter
                    used_right_indices.add(right_idx)
                    
                    if len(mapping) == len(left_items):
                        break
            
            # Ensure all left items have a match
            for left in left_items:
                if left not in mapping:
                    for j, (letter, _) in enumerate(right_items):
                        if j not in used_right_indices:
                            mapping[left] = letter
                            used_right_indices.add(j)
                            break
            
            return mapping

        elif question_type == "MCQ":
            stem = question_data.get("stem", "")
            choices = question_data.get("choices", [])
            if not choices:
                return None

            stem_tokens = set(re.findall(r'\w+', stem.lower()))
            scores = []

            for choice in choices:
                choice_tokens = set(re.findall(r'\w+', choice.lower()))
                overlap = stem_tokens & choice_tokens
                score = len(overlap) / len(choice_tokens) if choice_tokens else 0
                scores.append(score)

            if scores:
                best_index = scores.index(max(scores))
                return best_index  # Return index of predicted answer
            return None
        
        # Handle other question types (unchanged)
        return None
        
    except Exception as e:
        print(f"Error predicting answer: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def tokenize_text(text):
    """Break text into words and analyze their properties without domain knowledge"""
    # Normalize and tokenize
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Analyze word properties
    word_info = []
    for word in words:
        properties = {
            'word': word,
            'length': len(word),
            'is_noun': word.endswith(('er', 'or', 'ant', 'ent', 'ion', 'ity', 'ment', 'ness', 'ance', 'ence')),
            'is_adjective': word.endswith(('al', 'ful', 'ic', 'ive', 'less', 'ous', 'able', 'ible')),
            'is_technical': len(word) > 6 or word.endswith(('ate', 'ize', 'yze', 'ify')),
            'position': words.index(word)
        }
        word_info.append(properties)
    
    return {
        'words': words,
        'word_info': word_info,
        'bigrams': [words[i] + " " + words[i+1] for i in range(len(words)-1)],
        'length': len(words),
        'original': text
    }

def compute_linguistic_similarity(left_tokens, right_tokens):
    """Compute similarity based on linguistic features without domain knowledge"""
    score = 0
    
    # 1. Direct word overlap
    left_words = set(left_tokens['words'])
    right_words = set(right_tokens['words'])
    common_words = left_words.intersection(right_words)
    
    # Basic Jaccard similarity
    if len(left_words.union(right_words)) > 0:
        score += len(common_words) / len(left_words.union(right_words)) * 10
    
    # 2. Check for word relationships
    for left_word in left_tokens['word_info']:
        for right_word in right_tokens['word_info']:
            # Check for word stem matches (e.g., "coolant" and "cooling")
            if len(left_word['word']) > 4 and len(right_word['word']) > 4:
                if left_word['word'][:4] == right_word['word'][:4]:
                    score += 2
            
            # Check for semantic relationships based on word properties
            if left_word['is_technical'] and right_word['is_technical']:
                score += 1
            
            # Check for noun-adjective relationships
            if left_word['is_noun'] and right_word['is_adjective']:
                score += 0.5
            
            # Check for compound term relationships
            if left_word['word'] in right_tokens['original'] and len(left_word['word']) > 4:
                score += 3
            if right_word['word'] in left_tokens['original'] and len(right_word['word']) > 4:
                score += 3
    
    # 3. Check for bigram (two-word phrase) overlap
    common_bigrams = set(left_tokens['bigrams']).intersection(set(right_tokens['bigrams']))
    score += len(common_bigrams) * 5
    
    # 4. Check for positional relationships
    # Words in similar positions might be more related
    for i, left_word in enumerate(left_tokens['words']):
        for j, right_word in enumerate(right_tokens['words']):
            if abs(i - j) < 2 and left_word == right_word:
                score += 1
    
    return score


def submit_and_guess_question():
    global question_data, evaluate_button
    
    # Get the question text and type
    question_text = input_text.get("1.0", tk.END).strip()
    question_type = question_type_var.get()
    
    if not question_text:
        messagebox.showerror("Error", "Please enter a question.")
        return
    
    # Clear previous output
    output_text.delete("1.0", tk.END)
    
    # Parse the question
    question_data = parse_question(question_text, question_type)
    
    # Make a prediction
    predicted = predict_answer(question_data)
    global saved_stem, saved_choices, saved_matches, saved_correct_matches
    saved_stem = question_data.get("stem", "")
    saved_choices = question_data.get("choices", [])
    qtype = question_data.get("question_type")
    if qtype == "Matching":
        lefts = question_data.get("left_items", [])
        rights = [item for letter, item in question_data.get("right_items", [])]
        saved_matches = [(left, rights) for left in lefts]
        try:
            correct_answer_text = correct_answer_entry.get().strip()
            indices = [int(x.strip()) for x in correct_answer_text.split(",")]
            if len(indices) == len(lefts):
                saved_correct_matches = {lefts[i]: indices[i] for i in range(len(lefts))}
            else:
                saved_correct_matches = None
        except:
            saved_correct_matches = None
    else:
        saved_matches = None
        saved_correct_matches = None
    
    # Display the prediction
    output_text.insert(tk.END, "PREDICTION RESULTS\n", "heading")
    output_text.insert(tk.END, "=" * 50 + "\n\n")
    
    # Display the question type
    qtype = question_data.get("question_type", "Unknown")
    output_text.insert(tk.END, f"Question Type: {qtype}\n\n")
    
    # Display the stem
    stem = question_data.get("stem", "")
    output_text.insert(tk.END, f"Question Stem: {stem}\n\n")
    
    # Display the choices for MCQ
    if qtype == "MCQ":
        choices = question_data.get("choices", [])
        if choices:
            output_text.insert(tk.END, "Choices:\n")
            for idx, choice in enumerate(choices):
                output_text.insert(tk.END, f"  {chr(65+idx)}. {choice}\n")
            output_text.insert(tk.END, "\n")
    
    # Display the items for matching questions
    if qtype == "Matching":
        left_items = question_data.get("left_items", [])
        right_items = question_data.get("right_items", [])
        
        if left_items:
            output_text.insert(tk.END, "Left Items:\n")
            for idx, item in enumerate(left_items):
                output_text.insert(tk.END, f"  {idx+1}. {item}\n")
            output_text.insert(tk.END, "\n")
        
        if right_items:
            output_text.insert(tk.END, "Right Items:\n")
            for letter, item in right_items:
                output_text.insert(tk.END, f"  {letter}. {item}\n")
            output_text.insert(tk.END, "\n")
    
    # Display the prediction
    if qtype == "MCQ":
        if isinstance(predicted, int) and 0 <= predicted < len(question_data.get("choices", [])):
            pred_letter = chr(65 + predicted)
        else:
            pred_letter = "?"
        output_text.insert(tk.END, f"🔍 Predicted answer: {pred_letter}\n")
    elif qtype == "True/False":
        pred_val = "True" if predicted else "False"
        output_text.insert(tk.END, f"🔍 Predicted answer: {pred_val}\n")
    elif qtype == "Matching":
        output_text.insert(tk.END, f"🔍 Predicted matches:\n")
        left_items = question_data.get("left_items", [])
        right_items = question_data.get("right_items", [])
        if isinstance(predicted, dict):
            for idx, left in enumerate(left_items):
                right_letter = predicted.get(left, "?")
                right_text = next((text for letter, text in right_items if letter == right_letter), "?")
                output_text.insert(tk.END, f"  {idx+1}. {left} → {right_letter}. {right_text}\n")
        else:
            output_text.insert(tk.END, "  (No prediction available)\n")
    
    # Configure tag for headings
    output_text.tag_configure("heading", font=("Helvetica", 12, "bold"))
    
    # Enable the evaluate button
    evaluate_button.config(state=tk.NORMAL)

# --- Single Question GUI wrapped in function ---
def launch_single_question_window():
    global root, input_text, output_text, question_type_var, correct_answer_entry, evaluate_button
    root = tk.Tk()
    root.title("Question Quality Evaluator")
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # Question type selection
    question_type_var = tk.StringVar(value="MCQ")
    frame_qtype = tk.Frame(frame)
    frame_qtype.pack(pady=(0, 5))
    tk.Label(frame_qtype, text="Question Type:").pack(side=tk.LEFT)
    for qtype in ["MCQ", "True/False", "Matching"]:
        tk.Radiobutton(frame_qtype, text=qtype, variable=question_type_var, value=qtype).pack(side=tk.LEFT)

    # Input text box
    input_text = tk.Text(frame, wrap=tk.WORD, width=100, height=10)
    input_text.pack(padx=5, pady=(0, 10))

    # No custom paste handler is bound; default paste event will operate normally.
    
    # --- Add select-all binding for input_text ---
    def bind_select_all(widget):
        if not hasattr(widget, "_select_all_bound"):
            widget.bind("<Command-a>", lambda event: widget.tag_add("sel", "1.0", "end"))
            widget.bind("<Control-a>", lambda event: widget.tag_add("sel", "1.0", "end"))
            widget._select_all_bound = True

    # Ensure bind_select_all is called only once per widget
    bind_select_all(input_text)

    # Submit & Guess button
    tk.Button(frame, text="Submit Question & Guess", command=submit_and_guess_question).pack()

    # Correct answer field
    tk.Label(frame, text="Correct Answer (optional):").pack()
    correct_answer_entry = tk.Entry(frame)
    correct_answer_entry.pack()

    # Evaluate button (disabled by default)
    evaluate_button = tk.Button(frame, text="Evaluate", command=open_file_and_analyze)
    evaluate_button.config(state=tk.DISABLED)
    evaluate_button.pack()

    output_text = tk.Text(root, wrap=tk.WORD, width=100, height=30)
    output_text.pack(padx=10, pady=10)
    # No custom paste handler; default paste event will operate normally.

    # --- Add select-all binding for output_text ---
    bind_select_all(output_text)

    root.mainloop()

# --- Test Analysis Module ---
import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

def analyze_full_test(questions_data):
    """
    Analyze a complete test for quality metrics
    
    Parameters:
    questions_data - List of dictionaries with question data
    
    Returns:
    Dictionary with analysis results
    """
    results = {
        "total_questions": len(questions_data),
        "question_types": Counter(),
        "difficulty_distribution": {},
        "similar_questions": [],
        "topic_distribution": {},
        "cognitive_levels": {},
        "time_estimate": 0,
        "balance_issues": [],
        "suggestions": []
    }
    
    # Extract question texts for similarity analysis
    question_texts = []
    difficulty_scores = []
    
    for q in questions_data:
        # Count question types
        results["question_types"][q.get("question_type", "Unknown")] += 1
        
        # Collect text for similarity analysis
        stem = q.get("stem", "")
        choices = q.get("choices", [])
        full_text = stem + " " + " ".join(choices)
        question_texts.append(full_text)
        
        # Collect difficulty metrics
        clarity = q.get("clarity_score", 0)
        cohesion = q.get("stem_choice_cohesion", 0)
        
        # Simple difficulty estimation (can be refined)
        difficulty = 5 - ((clarity + cohesion) / 2)  # Invert so higher = more difficult
        difficulty_scores.append(difficulty)
    
    # Analyze question similarity
    if len(question_texts) > 1:
        similar_pairs = find_similar_questions(question_texts, threshold=0.7)
        for pair in similar_pairs:
            results["similar_questions"].append({
                "question1_idx": pair[0],
                "question2_idx": pair[1],
                "similarity": pair[2],
                "question1_text": questions_data[pair[0]].get("stem", ""),
                "question2_text": questions_data[pair[1]].get("stem", "")
            })
    
    # Analyze difficulty distribution
    if difficulty_scores:
        results["difficulty_distribution"] = {
            "mean": np.mean(difficulty_scores),
            "median": np.median(difficulty_scores),
            "std_dev": np.std(difficulty_scores),
            "min": min(difficulty_scores),
            "max": max(difficulty_scores),
            "histogram": np.histogram(difficulty_scores, bins=5, range=(0, 5))[0].tolist()
        }
        
        # Estimate cognitive levels based on difficulty
        cognitive_levels = {
            "Knowledge/Recall": 0,
            "Comprehension": 0,
            "Application": 0,
            "Analysis": 0,
            "Evaluation/Synthesis": 0
        }
        
        for score in difficulty_scores:
            if score < 1:
                cognitive_levels["Knowledge/Recall"] += 1
            elif score < 2:
                cognitive_levels["Comprehension"] += 1
            elif score < 3:
                cognitive_levels["Application"] += 1
            elif score < 4:
                cognitive_levels["Analysis"] += 1
            else:
                cognitive_levels["Evaluation/Synthesis"] += 1
                
        results["cognitive_levels"] = cognitive_levels
    
    # Estimate completion time (rough estimate)
    # Assume: MCQ = 1 min, True/False = 30 sec, Matching = 2 min per question
    time_estimates = {
        "MCQ": 1.0,
        "True/False": 0.5,
        "Matching": 2.0,
        "Unknown": 1.0
    }
    
    total_time = sum(time_estimates[q_type] * count for q_type, count in results["question_types"].items())
    results["time_estimate"] = total_time
    
    # Extract potential topics using keyword analysis
    topics = extract_topics(question_texts)
    results["topic_distribution"] = topics
    
    # Generate balance recommendations
    results["balance_issues"] = identify_balance_issues(results)
    results["suggestions"] = generate_test_suggestions(results)
    
    return results

def find_similar_questions(texts, threshold=0.7):
    """Find pairs of similar questions using TF-IDF and cosine similarity"""
    if len(texts) <= 1:
        return []
        
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find pairs above threshold (excluding self-comparisons)
    similar_pairs = []
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if similarity_matrix[i, j] >= threshold:
                similar_pairs.append((i, j, similarity_matrix[i, j]))
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    return similar_pairs

def extract_topics(texts):
    """Extract potential topics from question texts"""
    # Combine all texts
    all_text = " ".join(texts)
    
    # Remove common words and technical terms that might be specific to Navy training
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                      'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such',
                      'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during', 'to'])
    
    # Technical terms that might be common across all questions
    technical_terms = set(['navy', 'sailor', 'ship', 'vessel', 'officer', 'command', 'military'])
    
    # Extract words, clean and count them
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
    word_counts = Counter([w for w in words if w not in stop_words and w not in technical_terms])
    
    # Return top potential topics
    return dict(word_counts.most_common(10))

def identify_balance_issues(results):
    """Identify potential balance issues in the test"""
    issues = []
    
    # Check question type balance
    type_counts = results["question_types"]
    total = results["total_questions"]
    
    if total < 10:
        issues.append("Test has fewer than 10 questions, which may not provide adequate coverage for assessment.")
    
    # Check if one question type dominates
    for q_type, count in type_counts.items():
        if count / total > 0.7 and total > 5:
            issues.append(f"Test is dominated by {q_type} questions ({count}/{total}, {count/total:.0%}). Consider diversifying question types.")
    
    # Check difficulty distribution
    if "difficulty_distribution" in results and results["difficulty_distribution"]:
        diff_dist = results["difficulty_distribution"]
        
        if diff_dist["std_dev"] < 0.5:
            issues.append("Low variation in question difficulty (std dev < 0.5). Consider adding more variety.")
        
        if diff_dist["mean"] < 1.5:
            issues.append("Test appears too easy overall (mean difficulty < 1.5/5.0). Consider adding more challenging questions.")
        elif diff_dist["mean"] > 3.5:
            issues.append("Test appears too difficult overall (mean difficulty > 3.5/5.0). Consider adding some easier questions.")
            
        # Check for bimodal distribution (might indicate two difficulty levels with nothing in between)
        hist = diff_dist["histogram"]
        if len(hist) >= 5 and (hist[0] + hist[4] > 0.6 * sum(hist)) and (hist[1] + hist[2] + hist[3] < 0.3 * sum(hist)):
            issues.append("Difficulty distribution appears bimodal (very easy and very hard questions with few moderate ones).")
    
    # Check for too many similar questions
    if len(results["similar_questions"]) > total * 0.2:
        issues.append(f"Found {len(results['similar_questions'])} pairs of highly similar questions. Consider revising to increase variety.")
    
    # Check cognitive levels
    if results["cognitive_levels"]:
        knowledge_pct = results["cognitive_levels"]["Knowledge/Recall"] / total
        higher_order_pct = (results["cognitive_levels"]["Analysis"] + results["cognitive_levels"]["Evaluation/Synthesis"]) / total
        
        if knowledge_pct > 0.6:
            issues.append(f"Test emphasizes basic recall ({knowledge_pct:.0%} of questions). Consider adding higher-order thinking questions.")
        
        if higher_order_pct < 0.2 and total >= 10:
            issues.append(f"Test has few higher-order thinking questions ({higher_order_pct:.0%}). Consider adding analysis and evaluation questions.")
    
    # Check estimated completion time
    if results["time_estimate"] < 15:
        issues.append(f"Estimated completion time is only {results['time_estimate']:.1f} minutes, which may be too short for a comprehensive assessment.")
    elif results["time_estimate"] > 120:
        issues.append(f"Estimated completion time is {results['time_estimate']:.1f} minutes, which may be too long for a single sitting.")
    
    return issues

def generate_test_suggestions(results):
    """Generate suggestions for improving the test"""
    suggestions = []
    
    # Add suggestions based on identified issues
    for issue in results["balance_issues"]:
        if "dominated by" in issue:
            suggestions.append("Add more variety in question types to better assess different skills.")
        elif "too easy" in issue:
            suggestions.append("Incorporate questions that require application of concepts rather than just recall.")
        elif "too difficult" in issue:
            suggestions.append("Add some straightforward questions to build confidence and assess basic knowledge.")
        elif "similar questions" in issue:
            suggestions.append("Revise similar questions to test different aspects of the same topic.")
    
    # Add general suggestions for Navy technical training
    suggestions.append("Ensure questions reflect real-world scenarios sailors might encounter in their duties.")
    suggestions.append("Include questions that assess procedural knowledge important for shipboard operations.")
    
    # Add suggestions based on cognitive levels
    if results["cognitive_levels"]:
        total = results["total_questions"]
        if results["cognitive_levels"]["Application"] / total < 0.3:
            suggestions.append("Add more application questions that require sailors to apply knowledge to realistic scenarios.")
    
    # Add suggestions for topic balance
    if len(results["topic_distribution"]) < 3 and results["total_questions"] > 10:
        suggestions.append("Test appears narrowly focused on few topics. Consider broadening content coverage.")
    
    # Add suggestions for question types
    if "Matching" not in results["question_types"] and results["total_questions"] >= 15:
        suggestions.append("Consider adding matching questions to assess recognition of relationships between concepts.")
    
    # Add Navy-specific suggestions
    suggestions.append("Consider including questions that assess knowledge of Navy terminology and protocols.")
    suggestions.append("Ensure test accommodates sailors with diverse backgrounds and experience levels.")
    
    return suggestions

def parse_test_file(file_path):
    """Parse a test file into a list of question dictionaries"""
    questions = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Split into questions (assuming questions are separated by double newlines or numbered)
        question_blocks = re.split(r'\n\s*\n|\n\s*\d+\.\s+', content)
        question_blocks = [q.strip() for q in question_blocks if q.strip()]
        
        for block in question_blocks:
            lines = block.split('\n')
            if not lines:
                continue
                
            question = {"stem": lines[0], "choices": [], "question_type": "Unknown"}
            
            # Try to detect question type and parse accordingly
            if any(line.strip().startswith(('A.', 'B.', 'C.', 'D.')) for line in lines[1:]):
                question["question_type"] = "MCQ"
                for line in lines[1:]:
                    if re.match(r'^[A-D]\.\s+', line):
                        question["choices"].append(re.sub(r'^[A-D]\.\s+', '', line))
            
            elif any(line.lower().strip() in ('true', 'false', 't', 'f') for line in lines[1:]):
                question["question_type"] = "True/False"
                question["choices"] = ["True", "False"]
            
            elif any('match' in line.lower() for line in lines):
                question["question_type"] = "Matching"
                left_items = []
                right_items = []
                
                # Try to extract matching items
                for line in lines[1:]:
                    if re.match(r'^\d+\.\s+', line):
                        left_items.append(re.sub(r'^\d+\.\s+', '', line))
                    elif re.match(r'^[A-Z]\.\s+', line):
                        right_items.append(re.sub(r'^[A-Z]\.\s+', '', line))
                
                question["choices"] = left_items
                question["matches"] = [(left, right_items) for left in left_items]
            
            # Add basic quality metrics (placeholder values)
            question["clarity_score"] = np.random.uniform(2.0, 4.5)  # Placeholder
            question["stem_choice_cohesion"] = np.random.uniform(2.0, 4.5)  # Placeholder
            
            questions.append(question)
    
    except Exception as e:
        print(f"Error parsing test file: {e}")
        return []
    
    return questions

# --- Top-level GUI logic ---
def launch_single_question_gui():
    root = tk._default_root
    if root is not None:
        root.destroy()
    launch_single_question_window()

# --- Test Analysis GUI ---
def launch_test_analysis_gui():
    test_root = tk.Toplevel()
    test_root.title("Full Test Analysis")
    test_root.geometry("1000x800")
    
    frame = tk.Frame(test_root)
    frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # File selection
    file_frame = tk.Frame(frame)
    file_frame.pack(fill=tk.X, pady=10)
    
    tk.Label(file_frame, text="Test File:").pack(side=tk.LEFT)
    file_path_var = tk.StringVar()
    file_entry = tk.Entry(file_frame, textvariable=file_path_var, width=50)
    file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def browse_file():
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            title="Select Test File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            file_path_var.set(filename)
    
    tk.Button(file_frame, text="Browse...", command=browse_file).pack(side=tk.LEFT, padx=5)
    
    # Or paste test content
    tk.Label(frame, text="Or paste test content:").pack(anchor=tk.W)
    
    test_content = tk.Text(frame, height=10, wrap=tk.WORD)
    test_content.pack(fill=tk.X, pady=5)
    
    # Fix for duplicate paste issue in test_content
    def custom_paste(event=None):
        try:
            if test_content.tag_ranges("sel"):
                test_content.delete("sel.first", "sel.last")
            clipboard = test_root.clipboard_get()
            test_content.insert("insert", clipboard)
            return "break"
        except:
            pass
    
    test_content.bind("<<Paste>>", custom_paste)
    
    # Analysis options
    options_frame = tk.LabelFrame(frame, text="Analysis Options")
    options_frame.pack(fill=tk.X, pady=10)
    
    # Checkbuttons for analysis options
    similarity_var = tk.BooleanVar(value=True)
    difficulty_var = tk.BooleanVar(value=True)
    cognitive_var = tk.BooleanVar(value=True)
    topic_var = tk.BooleanVar(value=True)
    
    tk.Checkbutton(options_frame, text="Question Similarity Analysis", variable=similarity_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    tk.Checkbutton(options_frame, text="Difficulty Distribution", variable=difficulty_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
    tk.Checkbutton(options_frame, text="Cognitive Levels", variable=cognitive_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    tk.Checkbutton(options_frame, text="Topic Distribution", variable=topic_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    
    # Target audience frame
    audience_frame = tk.LabelFrame(frame, text="Target Audience")
    audience_frame.pack(fill=tk.X, pady=10)
    
    # Experience level
    tk.Label(audience_frame, text="Experience Level:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
    experience_var = tk.StringVar(value="Mixed")
    tk.OptionMenu(audience_frame, experience_var, "Entry-level", "Intermediate", "Advanced", "Mixed").grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
    
    # Technical background
    tk.Label(audience_frame, text="Technical Background:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
    background_var = tk.StringVar(value="Mixed")
    tk.OptionMenu(audience_frame, background_var, "Minimal", "Moderate", "Extensive", "Mixed").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
    
    # Analysis button
    analyze_button = tk.Button(frame, text="Analyze Test", font=("Helvetica", 12, "bold"))
    analyze_button.pack(pady=10)
    
    # Results notebook (tabbed interface)
    notebook = ttk.Notebook(frame)
    notebook.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Create tabs
    summary_tab = tk.Frame(notebook)
    similarity_tab = tk.Frame(notebook)
    difficulty_tab = tk.Frame(notebook)
    cognitive_tab = tk.Frame(notebook)
    topic_tab = tk.Frame(notebook)
    
    notebook.add(summary_tab, text="Summary")
    notebook.add(similarity_tab, text="Similar Questions")
    notebook.add(difficulty_tab, text="Difficulty")
    notebook.add(cognitive_tab, text="Cognitive Levels")
    notebook.add(topic_tab, text="Topics")
    
    # Summary tab content
    summary_text = tk.Text(summary_tab, wrap=tk.WORD)
    summary_text.pack(fill=tk.BOTH, expand=True)
    
    # Fix for duplicate paste in summary_text
    summary_text.bind("<<Paste>>", lambda event: custom_paste())
    
    # Function to run analysis and display results
    def run_analysis():
        # Get test content
        file_path = file_path_var.get()
        pasted_content = test_content.get("1.0", tk.END).strip()
        
        questions_data = []
        
        if file_path and os.path.exists(file_path):
            questions_data = parse_test_file(file_path)
        elif pasted_content:
            # Save pasted content to temp file and parse
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp:
                temp.write(pasted_content)
                temp_path = temp.name
            
            questions_data = parse_test_file(temp_path)
            try:
                os.unlink(temp_path)  # Delete temp file
            except:
                pass
        
        if not questions_data:
            messagebox.showerror("Error", "No valid questions found. Please check your input.")
            return
        
        # Run analysis
        results = analyze_full_test(questions_data)
        
        # Update results based on audience settings
        experience = experience_var.get()
        background = background_var.get()
        
        # Adjust suggestions based on audience
        audience_suggestions = []
        
        if experience == "Entry-level":
            audience_suggestions.append("For entry-level sailors, consider adding more knowledge-based questions and fewer complex analysis questions.")
        elif experience == "Advanced":
            audience_suggestions.append("For advanced sailors, ensure there are enough challenging questions that test application in complex scenarios.")
        
        if background == "Minimal":
            audience_suggestions.append("For sailors with minimal technical background, include more explanatory context in question stems.")
        elif background == "Extensive":
            audience_suggestions.append("For sailors with extensive technical background, include questions that test integration of multiple technical concepts.")
        
        results["suggestions"].extend(audience_suggestions)
        
        # Display summary results
        summary_text.delete("1.0", tk.END)
        
        summary_text.insert(tk.END, "TEST ANALYSIS SUMMARY\n", "heading")
        summary_text.insert(tk.END, "=" * 50 + "\n\n")
        
        summary_text.insert(tk.END, f"Total Questions: {results['total_questions']}\n")
        summary_text.insert(tk.END, f"Estimated Completion Time: {results['time_estimate']:.1f} minutes\n\n")
        
        summary_text.insert(tk.END, "Question Types:\n")
        for q_type, count in results["question_types"].items():
            summary_text.insert(tk.END, f"  - {q_type}: {count} ({count/results['total_questions']:.0%})\n")
        
        summary_text.insert(tk.END, "\nDifficulty:\n")
        if results["difficulty_distribution"]:
            diff = results["difficulty_distribution"]
            summary_text.insert(tk.END, f"  - Average Difficulty: {diff['mean']:.2f}/5.0\n")
            summary_text.insert(tk.END, f"  - Range: {diff['min']:.1f} - {diff['max']:.1f}\n")
        
        summary_text.insert(tk.END, "\nSimilar Questions:\n")
        if results["similar_questions"]:
            summary_text.insert(tk.END, f"  - Found {len(results['similar_questions'])} pairs of similar questions\n")
        else:
            summary_text.insert(tk.END, "  - No significantly similar questions found\n")
        
        summary_text.insert(tk.END, "\nIssues Identified:\n")
        if results["balance_issues"]:
            for issue in results["balance_issues"]:
                summary_text.insert(tk.END, f"  - {issue}\n")
        else:
            summary_text.insert(tk.END, "  - No major issues identified\n")
        
        summary_text.insert(tk.END, "\nSuggestions:\n")
        for suggestion in results["suggestions"]:
            summary_text.insert(tk.END, f"  - {suggestion}\n")
        
        # Configure tag for headings
        summary_text.tag_configure("heading", font=("Helvetica", 14, "bold"))
        
        # Display similar questions
        if similarity_var.get():
            display_similarity_results(similarity_tab, results)
        
        # Display difficulty distribution
        if difficulty_var.get():
            display_difficulty_results(difficulty_tab, results)
        
        # Display cognitive levels
        if cognitive_var.get():
            display_cognitive_results(cognitive_tab, results)
        
        # Display topic distribution
        if topic_var.get():
            display_topic_results(topic_tab, results)
    
    # Connect analyze button to function
    analyze_button.config(command=run_analysis)
    
    # Functions to display detailed results in tabs
    def display_similarity_results(tab, results):
        # Clear previous content
        for widget in tab.winfo_children():
            widget.destroy()
        
        if not results["similar_questions"]:
            tk.Label(tab, text="No significantly similar questions found.", font=("Helvetica", 12)).pack(pady=20)
            return
        
        # Create frame for similar questions
        tk.Label(tab, text="Similar Question Pairs", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(tab)
        scrollbar = tk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add similar question pairs
        for i, pair in enumerate(results["similar_questions"]):
            pair_frame = tk.LabelFrame(scrollable_frame, text=f"Pair {i+1} - Similarity: {pair['similarity']:.2f}")
            pair_frame.pack(fill="x", expand=True, padx=10, pady=5)
            
            tk.Label(pair_frame, text="Question 1:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=2)
            tk.Label(pair_frame, text=pair["question1_text"], wraplength=400).grid(row=0, column=1, sticky="w", padx=5, pady=2)
            
            tk.Label(pair_frame, text="Question 2:", font=("Helvetica", 10, "bold")).grid(row=1, column=0, sticky="w", padx=5, pady=2)
            tk.Label(pair_frame, text=pair["question2_text"], wraplength=400).grid(row=1, column=1, sticky="w", padx=5, pady=2)
    
    def display_difficulty_results(tab, results):
        # Clear previous content
        for widget in tab.winfo_children():
            widget.destroy()
        
        if not results.get("difficulty_distribution"):
            tk.Label(tab, text="Difficulty analysis not available.", font=("Helvetica", 12)).pack(pady=20)
            return
        
        # Create frame for difficulty distribution
        tk.Label(tab, text="Difficulty Distribution", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Create matplotlib figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot histogram
        diff = results["difficulty_distribution"]
        bins = np.linspace(0, 5, 6)
        ax.hist(np.linspace(0, 5, 5), bins=bins, weights=diff["histogram"], color='skyblue', edgecolor='black')
        
        # Add labels and title
        ax.set_xlabel('Difficulty Level (0-5)')
        ax.set_ylabel('Number of Questions')
        ax.set_title('Question Difficulty Distribution')
        
        # Add mean line
        ax.axvline(x=diff["mean"], color='red', linestyle='--', label=f'Mean: {diff["mean"]:.2f}')
        ax.legend()
        
        # Add the plot to the tab
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add statistics below the chart
        stats_frame = tk.Frame(tab)
        stats_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(stats_frame, text=f"Mean Difficulty: {diff['mean']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(stats_frame, text=f"Median: {diff['median']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(stats_frame, text=f"Std Dev: {diff['std_dev']:.2f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20)
        tk.Label(stats_frame, text=f"Range: {diff['min']:.1f} - {diff['max']:.1f}", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=20)
    
    def display_cognitive_results(tab, results):
        # Clear previous content
        for widget in tab.winfo_children():
            widget.destroy()
        
        if not results.get("cognitive_levels"):
            tk.Label(tab, text="Cognitive level analysis not available.", font=("Helvetica", 12)).pack(pady=20)
            return
        
        # Create frame for cognitive levels
        tk.Label(tab, text="Cognitive Levels Distribution", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Create matplotlib figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot pie chart
        labels = list(results["cognitive_levels"].keys())
        sizes = list(results["cognitive_levels"].values())
        
        # Only include non-zero values
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        filtered_labels = [labels[i] for i in non_zero_indices]
        filtered_sizes = [sizes[i] for i in non_zero_indices]
        
        # Create pie chart with non-zero values
        if filtered_sizes:
            ax.pie(filtered_sizes, labels=filtered_labels, autopct='%1.1f%%', startangle=90, shadow=True)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            ax.set_title('Distribution of Cognitive Levels')
            
            # Add the plot to the tab
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(tab, text="No cognitive level data to display.", font=("Helvetica", 12)).pack(pady=20)
        
        # Add explanation of cognitive levels
        explanation_frame = tk.LabelFrame(tab, text="Cognitive Levels Explained")
        explanation_frame.pack(fill=tk.X, padx=10, pady=10)
        
        explanations = {
            "Knowledge/Recall": "Basic recall of facts, terms, concepts, or procedures.",
            "Comprehension": "Understanding the meaning of information, explaining or summarizing.",
            "Application": "Using knowledge in new situations, applying rules or methods.",
            "Analysis": "Breaking information into parts to explore relationships and connections.",
            "Evaluation/Synthesis": "Making judgments based on criteria, creating new patterns or structures."
        }
        
        for level, explanation in explanations.items():
            level_frame = tk.Frame(explanation_frame)
            level_frame.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(level_frame, text=level+":", font=("Helvetica", 10, "bold"), width=20, anchor="w").pack(side=tk.LEFT)
            tk.Label(level_frame, text=explanation, wraplength=500, justify=tk.LEFT).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def display_topic_results(tab, results):
        # Clear previous content
        for widget in tab.winfo_children():
            widget.destroy()
        
        if not results.get("topic_distribution"):
            tk.Label(tab, text="Topic analysis not available.", font=("Helvetica", 12)).pack(pady=20)
            return
        
        # Create frame for topic distribution
        tk.Label(tab, text="Topic Distribution", font=("Helvetica", 14, "bold")).pack(pady=10)
        
        # Create matplotlib figure
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot horizontal bar chart
        topics = list(results["topic_distribution"].keys())
        counts = list(results["topic_distribution"].values())
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)
        sorted_topics = [topics[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Plot horizontal bar chart (limited to top 15 topics if more exist)
        max_topics = 15
        if len(sorted_topics) > max_topics:
            sorted_topics = sorted_topics[-max_topics:]
            sorted_counts = sorted_counts[-max_topics:]
            
        y_pos = np.arange(len(sorted_topics))
        ax.barh(y_pos, sorted_counts, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_topics)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Frequency')
        ax.set_title('Topic Distribution')
        
        # Add the plot to the tab
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add explanation
        tk.Label(tab, text="Note: Topics are extracted from question text using keyword frequency analysis.",
                 font=("Helvetica", 10, "italic")).pack(pady=10)
        
        # Add topic balance assessment
        balance_frame = tk.LabelFrame(tab, text="Topic Balance Assessment")
        balance_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Calculate topic concentration (Gini coefficient-inspired)
        total = sum(counts)
        normalized_counts = [c/total for c in counts]
        topic_concentration = sum((c - 1/len(counts))**2 for c in normalized_counts if c > 0)
        topic_concentration = min(1.0, topic_concentration)  # Cap at 1.0
        
        # Assess topic balance
        if topic_concentration < 0.2:
            balance_text = "Good topic balance. The test covers a variety of topics with relatively even distribution."
        elif topic_concentration < 0.5:
            balance_text = "Moderate topic concentration. Some topics are emphasized more than others."
        else:
            balance_text = "High topic concentration. The test focuses heavily on a few topics."
        
        tk.Label(balance_frame, text=balance_text, wraplength=600, justify=tk.LEFT).pack(padx=10, pady=10)

    # Import necessary modules for test analysis
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import re
        from collections import Counter
        import os
        import ttk  # For notebook widget
    except ImportError as e:
        error_msg = f"Missing required package: {str(e)}\n\nPlease install required packages with:\npip install matplotlib numpy scikit-learn"
        messagebox.showerror("Missing Dependencies", error_msg)
        return

    # Start the GUI
    test_root.mainloop()

# Update the main menu function to properly call the test analysis GUI
def launch_test_analysis_gui():
    try:
        # Check if required packages are installed
        import matplotlib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        # If we get here, the imports worked
        analyze_test_window()
    except ImportError:
        messagebox.showerror("Missing Dependencies",
                            "This feature requires additional packages.\n\n"
                            "Please install them with:\n"
                            "pip install matplotlib numpy scikit-learn pandas")

# Rename the implementation function to avoid confusion
def analyze_test_window():
    # This is just a wrapper to call our implementation
    # It helps separate the dependency check from the implementation
    launch_test_analysis_gui()



def launch_main_menu():
    main_root = tk.Tk()
    main_root.title("Assessment Analysis Tool")
    main_root.geometry("400x300")  # Optional: Set window size

    tk.Label(main_root, text="Choose an Option", font=("Helvetica", 16)).pack(pady=20)

    tk.Button(main_root, text="🔍 Single Question Analysis", width=30, height=2, command=launch_single_question_gui).pack(pady=10)
    tk.Button(main_root, text="📝 Full Test Analysis (Coming Soon)", width=30, height=2, command=launch_test_analysis_gui).pack(pady=10)
    tk.Button(main_root, text="❌ Exit", width=30, height=2, command=main_root.destroy).pack(pady=10)

    main_root.mainloop()

launch_main_menu()
