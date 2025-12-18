# Issue #11: Add Visual Grouping for Related Chat Messages

**GitHub Issue:** https://github.com/ThriveAI-Solutions/thrive_ui/issues/11
**BD Issue:** thrive_ui-7u9

## Problem Summary

Currently, chat messages (question, SQL, data table, charts, summary, follow-up questions) display as a flat list, making it difficult to identify which outputs belong to which question.

## Existing Infrastructure

The codebase already has `group_id` support:
- `generate_group_id()` - Creates UUID for message flows (`utils/chat_bot_helper.py:489-491`)
- `get_current_group_id()` - Gets current group ID from session state (`utils/chat_bot_helper.py:494-498`)
- `start_new_group()` - Starts new message group (`utils/chat_bot_helper.py:501-504`)
- `Message.group_id` - Column in the Message model (`orm/models.py:124`)
- All messages in a Q&A flow already receive the same `group_id`

## Current Rendering Flow

1. `views/chat_bot.py:229-235` - Iterates through messages and calls `render_message()`
2. `utils/chat_bot_helper.py:771-774` - `render_message()` wraps each message in `st.chat_message()`

## Implementation Plan

### Step 1: Create Group Rendering Utility
Create a helper function to group consecutive messages by `group_id` and render them within styled containers.

**File:** `utils/chat_bot_helper.py`
- Add function `group_messages_by_id(messages)` that returns list of (group_id, [messages]) tuples
- Add function `render_message_group(messages, group_index)` that renders a group with visual styling

### Step 2: Add CSS Styling for Message Groups
Add CSS styling using Streamlit's `st.markdown()` with `unsafe_allow_html=True` for visual grouping.

**Styling approach:**
- Left border accent (subtle colored vertical line)
- Slight background tint for alternating groups
- Small margin/padding between groups
- Theme-aware colors (works with existing ThemeType enum)

### Step 3: Update Message Rendering Loop
Modify `views/chat_bot.py` to use the new group rendering.

**Changes:**
- Group messages by `group_id` before rendering
- Render each group within a styled container
- Handle messages with no `group_id` (legacy messages) gracefully

### Step 4: Handle Edge Cases
- Messages without `group_id` (legacy data) - render individually without grouping
- Single-message groups - still wrap in container for consistency
- Empty groups - skip rendering

### Step 5: Add Unit Tests
- Test `group_messages_by_id()` function
- Test rendering with multiple groups
- Test backward compatibility with messages without `group_id`

## Visual Design

```
┌─────────────────────────────────────────┐
│ [User Message] What is the avg patient age?
│ [Assistant] Here's the SQL...
│ [Assistant] [Data Table]
│ [Assistant] [Chart]
│ [Assistant] Summary text...
│ [Assistant] Follow-up questions...
└─────────────────────────────────────────┘
   (subtle gap)
┌─────────────────────────────────────────┐
│ [User Message] How many patients visited?
│ [Assistant] Here's the SQL...
│ [Assistant] [Data Table]
│ [Assistant] Summary text...
└─────────────────────────────────────────┘
```

**CSS Styling:**
- Left border: 3px solid with theme-aware color
- Background: Very subtle tint (rgba with low opacity)
- Padding: Small internal padding
- Margin: Bottom margin between groups
- Border radius: Subtle rounding

## Files to Modify

1. `utils/chat_bot_helper.py` - Add grouping functions
2. `views/chat_bot.py` - Update rendering loop (lines 229-235)
3. `tests/views/test_chat_bot.py` - Add tests for grouping functionality

## Acceptance Criteria

- [x] Related messages share a visual container
- [x] Container has clear visual boundary (left border + subtle background)
- [x] Each question starts a new visual group
- [x] Groups are visually distinct from each other
- [x] Works with all message types (SQL, tables, charts, summaries, follow-ups)
- [x] Works with both light and dark themes
- [x] Backward compatible with messages without `group_id`
- [x] Does not add noticeable latency to rendering
