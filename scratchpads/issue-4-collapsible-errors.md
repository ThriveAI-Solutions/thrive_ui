# Issue #4: Improve Error Handling UX with Collapsible Error Details

**Link:** https://github.com/ThriveAI-Solutions/thrive_ui/issues/4
**bd Issue:** thrive_ui-zqg

## Problem Summary

Error messages in the chatbot display as large red blocks that are visually overwhelming and take up significant screen space. Users cannot collapse or hide error details.

## Affected Files

1. **`utils/chat_bot_helper.py:576`** - `_render_error()` function for MessageType.ERROR messages
2. **`utils/chat_bot_helper.py:1219-1232`** - Inline SQL execution error display
3. **`views/chat_bot.py:261-273`** - Persistent error panel between reruns

## Implementation Plan

### Step 1: Update `_render_error()` function (chat_bot_helper.py:573-576)
- Replace plain `st.error(message.content)` with a warning + collapsible expander pattern
- Show brief user-friendly message by default
- Put full technical details inside a collapsed expander

### Step 2: Update inline SQL execution error (chat_bot_helper.py:1219-1232)
- Change `st.error()` to `st.warning()` for less harsh visuals
- Move database error details into a collapsible expander
- Keep retry and show SQL buttons visible and accessible

### Step 3: Update persistent error panel (views/chat_bot.py:261-273)
- Apply same collapsible pattern to the persistent error panel
- Ensure retry and show SQL buttons remain accessible outside the expander

## Design Pattern

```python
# Instead of:
st.error(message.content)

# Use:
st.warning("An error occurred while processing your request.")
with st.expander("View error details", expanded=False):
    st.code(message.content, language="text")
```

For SQL execution errors:
```python
# Brief, non-intrusive message
st.warning("I couldn't execute the generated SQL.")
with st.expander("View error details", expanded=False):
    if error_msg:
        st.markdown(f"**Database error:** {error_msg}")
# Buttons remain outside expander for easy access
```

## Testing Plan

1. Trigger an error by asking a question that generates invalid SQL
2. Verify error message is collapsed by default
3. Verify error details can be expanded
4. Verify retry and show SQL buttons work correctly
5. Run existing tests to ensure no regressions
