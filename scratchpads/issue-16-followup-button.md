# Issue #16: Add Follow-up Button to Message Groups in Chat Interface

**GitHub Issue:** https://github.com/ThriveAI-Solutions/thrive_ui/issues/16
**BD Issue:** thrive_ui-ofy

## Problem Summary

After the implementation of visual message grouping (issue #11), users have clear visual distinction between Q&A conversation groups. However, users may not be aware of the powerful `/followup` magic commands available to further analyze their query results. A contextual follow-up button will increase discoverability.

## Existing Infrastructure

From my analysis of the codebase:

1. **Message Grouping** (`utils/chat_bot_helper.py`):
   - `group_messages_by_id()` - Groups messages by `group_id` (lines 548-585)
   - `render_message_group()` - Renders grouped messages in a styled container (lines 616-637)
   - `get_message_group_css()` - Provides CSS styling for containers (lines 588-613)

2. **Follow-up Commands** (`utils/magic_functions.py`):
   - `FOLLOW_UP_MAGIC_RENDERERS` - Dictionary of follow-up commands (line 5077+)
   - Categories: Data Exploration, Data Quality, Statistical Analysis, Visualizations, Machine Learning
   - Includes: `describe`, `profile`, `heatmap`, `clusters`, `pca`, `missing`, `duplicates`, etc.

3. **Rendering Flow** (`views/chat_bot.py`):
   - Messages are grouped and rendered via `render_message_group()` (lines 244-251)
   - CSS is injected at the top of the messages container (line 242)

## Implementation Plan

### Step 1: Add Follow-up Button Rendering Logic
Modify `render_message_group()` to accept additional parameters for showing the follow-up button on the last group only.

### Step 2: Add CSS Styling for the Follow-up Button
Create CSS that positions the button overlapping the bottom border.

### Step 3: Create Follow-up Command Popover
Create a popover that shows available follow-up commands when the button is clicked.

### Step 4: Update Chat Bot View
Modify `views/chat_bot.py` to pass `is_last_group` to `render_message_group()`.

### Step 5: Add Helper to Detect Data Results
Create logic to determine if a message group has data results that support follow-up commands.

### Step 6: Add Tests
Write unit tests for the new functionality.

## Acceptance Criteria

- [ ] Button appears only on the last message group
- [ ] Button is visually integrated with the group border design (overlaps bottom)
- [ ] Clicking shows available follow-up commands in a popover
- [ ] Commands are contextually relevant to the data
- [ ] Commands can be selected to execute
- [ ] Button doesn't appear for groups without data results
- [ ] Works with existing theme system (HealtheLink theme colors)
- [ ] Keyboard accessible
- [ ] No noticeable performance impact

## Files to Modify

1. `utils/chat_bot_helper.py` - Add button rendering and CSS
2. `views/chat_bot.py` - Pass `is_last_group` flag
3. `tests/utils/test_chat_bot_helper.py` - Add tests
