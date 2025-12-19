# Issue #22: Move Follow-up Questions to Follow Up Popover

**GitHub Issue:** https://github.com/ThriveAI-Solutions/thrive_ui/issues/22
**BD Issue:** thrive_ui-huw

## Problem Summary

Currently there are two separate popovers in the message UI:
1. **Actions popover** - Contains "Speak Summary", "Follow-up Questions", "Generate Table", chart buttons, SQL display
2. **Follow Up popover** - Contains quick analysis commands (describe, profile, heatmap, etc.)

This creates redundancy:
- "Follow-up Questions" logically belongs with other follow-up functionality
- Actions button appears on all message groups, even when not useful
- The Follow Up button already only appears on the last visible group

## Implementation Plan

### Step 1: Update `render_followup_button()` to Include Follow-up Questions
- Add a new "AI Questions" category to the Follow Up popover
- Store the group's message data in session state so we can access it for generating questions
- Add a button that calls `get_followup_questions()` with the stored context

### Step 2: Remove "Follow-up Questions" from Actions Popover
- Remove the "Follow-up Questions" button from `_render_summary_actions_popover()`
- Keep all other functionality intact (Speak Summary, Generate Table, Charts, SQL)

### Step 3: Conditionally Show Actions Popover Only on Last Group
- Modify `_render_summary()` to accept an `is_last_group` parameter
- Only render the Actions popover when `is_last_group=True`
- Update `render_message()` to pass this context through

### Step 4: Update `render_message_group()` to Pass Context
- Modify the rendering flow to pass `is_last_group` through to `_render_summary()`
- This requires updating the message rendering pipeline

### Step 5: Write Tests
- Add unit tests for the new behavior

## Key Files to Modify

1. `utils/chat_bot_helper.py`:
   - `get_followup_command_suggestions()` - Add AI Questions category
   - `render_followup_button()` - Store message context, add AI questions button
   - `_render_summary_actions_popover()` - Remove Follow-up Questions button
   - `_render_summary()` - Add is_last_group parameter
   - `render_message()` - Pass is_last_group context
   - `render_message_group()` - Thread is_last_group through rendering

## Acceptance Criteria

- [x] Follow Up popover has "AI Questions" category with "Generate" button
- [x] Clicking "Generate" calls AI to generate follow-up questions
- [x] Actions popover no longer has "Follow-up Questions" button
- [x] Actions popover only appears on the last message group
- [x] All existing Actions functionality still works (Speak, Table, Charts, SQL)
- [x] Follow Up button behavior unchanged (only on last group with data)

## Implementation Summary

### Changes Made

1. **`render_followup_button()`** - Added `messages` parameter to pass group context
   - Finds SUMMARY message in the group to get question/SQL/DataFrame context
   - Added "💡 AI Questions" section with "Generate" button
   - Button calls `get_followup_questions()` with the stored context

2. **`_render_summary_actions_popover()`** - Removed "Follow-up Questions" button
   - Added comment explaining the button was moved to Follow Up popover

3. **`_render_summary()`** - Added conditional rendering based on `is_last_group`
   - Reads `_render_is_last_group` from session state
   - Only renders Actions popover when `is_last_group=True`
   - Adjusts column layout based on whether Actions is shown

4. **`render_message_group()`** - Sets context for child renderers
   - Sets `st.session_state["_render_is_last_group"]` before rendering messages
   - Passes `messages` list to `render_followup_button()` for AI Questions

### Tests Added

- `TestRenderFollowupButtonWithAIQuestions` - Verifies new `messages` parameter
- `TestRenderSummaryActionsPopover` - Verifies Follow-up Questions button removed
- `TestRenderSummaryIsLastGroupContext` - Verifies is_last_group context passing
- Updated `test_render_summary_message` to mock `_render_is_last_group`
