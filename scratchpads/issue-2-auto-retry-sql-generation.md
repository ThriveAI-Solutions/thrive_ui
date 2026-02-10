# Issue #2: Auto-retry SQL generation on failure with progressive error context

GitHub Issue: https://github.com/ThriveAI-Solutions/thrive_ui/issues/2

## Summary

Add automatic retry logic (up to 2 attempts) when SQL generation fails, with progressive error context feedback to help the LLM generate more accurate SQL on each retry.

## Current State Analysis

### Key Files
1. **`utils/chat_bot_helper.py`** - `normal_message_flow()` function (lines 785-1134)
   - Currently has manual retry logic (lines 950-980) that requires user to click "Retry" button
   - Stores error context in session state (`last_run_sql_error`, `last_failed_sql`)
   - Uses `use_retry_context` flag to trigger retry with context

2. **`utils/vanna_calls.py`** - `generate_sql_retry()` method (lines 768-788)
   - Existing method that accepts `question`, `failed_sql`, and `error_message`
   - Builds an augmented question with error context
   - Calls the regular `generate_sql()` method with enhanced prompt

3. **`utils/config_helper.py`** - Configuration utilities
   - Pattern for reading config from env vars, secrets.toml, or defaults

### Current Flow
1. User asks question
2. `generate_sql()` generates SQL
3. `run_sql()` executes SQL
4. If execution fails:
   - Error stored in session state
   - Manual "Retry" button shown
   - User must click to retry
   - `generate_sql_retry()` called with error context

## Implementation Plan

### Step 1: Add retry configuration to config_helper.py
- Add `get_max_sql_retries()` function to read from:
  1. ENV: `MAX_SQL_RETRIES`
  2. secrets.toml: `[retry] max_sql_retries`
  3. Default: 2

### Step 2: Enhance generate_sql_retry in vanna_calls.py
- Add `attempt_number` parameter (1-based, so attempt 2 means first retry)
- Create progressive guidance messages based on attempt number:
  - Attempt 2: "Try a DIFFERENT approach. Consider different JOINs, subqueries, or alternative columns."
  - Attempt 3: "Try the SIMPLEST possible query. Remove JOINs, use only essential columns, break into smaller queries."

### Step 3: Implement auto-retry loop in chat_bot_helper.py
- Wrap SQL generation and execution in a retry loop
- Show visual feedback during retries (e.g., "Attempt 2/3: Trying a different approach...")
- Track retry attempts and exit loop on success or max retries
- Only show manual retry button after automatic retries are exhausted

### Step 4: Add non-recoverable error detection
- Detect errors that won't benefit from retries:
  - "table not found" / "relation does not exist"
  - "column not found" / "column does not exist"
  - Permission errors
- Skip automatic retries for these errors

### Step 5: Add logging for retry analytics
- Log all retry attempts with:
  - Original question
  - Attempt number
  - Failed SQL
  - Error message
  - Whether retry succeeded

### Step 6: Write unit tests
- Test retry logic triggers on SQL execution failure
- Test retry count limits (max 2 retries)
- Test progressive error context is passed correctly
- Test successful retry short-circuits remaining attempts
- Test non-recoverable error detection skips retries

## Non-Recoverable Error Patterns

```python
NON_RECOVERABLE_PATTERNS = [
    r"relation.*does not exist",
    r"table.*does not exist",
    r"column.*does not exist",
    r"permission denied",
    r"access denied",
    r"authentication failed",
]
```

## Progressive Retry Prompts

```python
RETRY_GUIDANCE = {
    2: (
        "The previous SQL failed. Please try a DIFFERENT approach to answer this question. "
        "Consider using different JOINs, subqueries, or alternative column selections."
    ),
    3: (
        "Multiple attempts have failed. Please try the SIMPLEST possible query that could answer this question. "
        "Consider: removing JOINs, using only essential columns, or breaking into smaller queries."
    ),
}
```

## Testing Strategy

1. **Unit tests**: Mock VannaService methods, verify retry logic
2. **Integration tests**: Mock SQL execution failures, test end-to-end
3. **Manual testing**: Use Puppeteer to verify UI feedback during retries
