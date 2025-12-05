# StatsBomb Euro 2020 SQLite Database Specification

## Overview

This document provides a comprehensive specification for the StatsBomb Euro 2020 SQLite database (`statsbomb_euro2020.db`), which contains all football/soccer data for the UEFA Euro 2020 tournament in a normalized, queryable format.

### Database Details

- **Format**: SQLite 3
- **Size**: 844.5 MB
- **Records**: 2,733,668 total records
- **Tournament**: UEFA Euro 2020 (held in 2021)
- **Data Version**: StatsBomb Open Data v1.1.x
- **License**: CC BY-NC 4.0

## Database Schema

### Core Tables

#### 1. `matches` - Tournament Matches

**Purpose**: Metadata for all 51 UEFA Euro 2020 matches

| Column                  | Type    | Constraints | Description                  | Example                            |
| ----------------------- | ------- | ----------- | ---------------------------- | ---------------------------------- |
| `match_id`              | INTEGER | PRIMARY KEY | Unique match identifier      | 3788741                            |
| `match_date`            | TEXT    | NOT NULL    | Match date (YYYY-MM-DD)      | "2021-06-11"                       |
| `kick_off`              | TEXT    |             | Kick-off time (HH:MM:SS.sss) | "21:00:00.000"                     |
| `competition`           | TEXT    |             | Competition name             | "Europe - UEFA Euro"               |
| `season`                | TEXT    |             | Season identifier            | "2020"                             |
| `home_team`             | TEXT    | NOT NULL    | Home team name               | "Switzerland"                      |
| `away_team`             | TEXT    | NOT NULL    | Away team name               | "Turkey"                           |
| `home_score`            | INTEGER |             | Final home team score        | 3                                  |
| `away_score`            | INTEGER |             | Final away team score        | 1                                  |
| `match_status`          | TEXT    |             | Data availability status     | "available"                        |
| `match_status_360`      | TEXT    |             | 360° data availability       | "available"                        |
| `last_updated`          | TEXT    |             | Last data update timestamp   | "2023-07-07T12:07:59.867"          |
| `last_updated_360`      | TEXT    |             | Last 360° data update        | "2023-07-07T12:08:02.707"          |
| `match_week`            | INTEGER |             | Tournament week number       | 1                                  |
| `competition_stage`     | TEXT    |             | Tournament stage             | "Group Stage", "Round of 16"       |
| `stadium`               | TEXT    |             | Stadium name                 | "Saint-Petersburg Stadium"         |
| `referee`               | TEXT    |             | Referee name                 | "Michael Oliver"                   |
| `home_managers`         | TEXT    | JSON        | Home team managers (JSON)    | `[{"id": 123, "name": "Manager"}]` |
| `away_managers`         | TEXT    | JSON        | Away team managers (JSON)    | `[{"id": 456, "name": "Manager"}]` |
| `data_version`          | TEXT    |             | StatsBomb data version       | "1.1.0"                            |
| `shot_fidelity_version` | TEXT    |             | Shot data version            | "2"                                |
| `xy_fidelity_version`   | TEXT    |             | Position data version        | "2"                                |

**Row Count**: 51 rows (one per match)

#### 2. `events` - Match Events

**Purpose**: Detailed event data for all matches (passes, shots, tackles, etc.)

##### Core Event Fields

| Column               | Type    | Constraints | Description                                 |
| -------------------- | ------- | ----------- | ------------------------------------------- |
| `id`                 | TEXT    | PRIMARY KEY | Unique event identifier                     |
| `index_num`          | INTEGER |             | Event sequence number in match              |
| `match_id`           | INTEGER | FOREIGN KEY | References matches(match_id)                |
| `period`             | INTEGER |             | Match period (1=1st half, 2=2nd half, etc.) |
| `minute`             | INTEGER |             | Minute of occurrence                        |
| `second`             | INTEGER |             | Second of occurrence                        |
| `timestamp`          | TEXT    |             | Precise timestamp (MM:SS.sss)               |
| `type`               | TEXT    |             | Event type (Pass, Shot, Tackle, etc.)       |
| `possession`         | INTEGER |             | Possession sequence number                  |
| `possession_team`    | TEXT    |             | Team in possession                          |
| `possession_team_id` | INTEGER |             | Team ID in possession                       |
| `play_pattern`       | TEXT    |             | How play started                            |
| `team`               | TEXT    |             | Team performing action                      |
| `team_id`            | INTEGER |             | Team ID performing action                   |
| `player`             | TEXT    |             | Player name                                 |
| `player_id`          | REAL    |             | Player ID                                   |
| `position`           | TEXT    |             | Player position                             |
| `duration`           | REAL    |             | Event duration (seconds)                    |
| `location`           | TEXT    | JSON        | Event coordinates [x, y]                    |

##### Pass-Specific Fields

| Column                  | Type        | Description                                        |
| ----------------------- | ----------- | -------------------------------------------------- |
| `pass_end_location`     | TEXT (JSON) | Pass destination [x, y]                            |
| `pass_length`           | REAL        | Pass distance (yards)                              |
| `pass_angle`            | REAL        | Pass angle (radians)                               |
| `pass_height`           | TEXT        | Pass height (Ground Pass, High Pass, Low Pass)     |
| `pass_body_part`        | TEXT        | Body part used (Left Foot, Right Foot, Head, etc.) |
| `pass_type`             | TEXT        | Pass type (Corner, Throw-in, Free Kick, etc.)      |
| `pass_outcome`          | TEXT        | Pass result (Incomplete, Out, etc.)                |
| `pass_technique`        | TEXT        | Pass technique (Inswinging, Outswinging, etc.)     |
| `pass_recipient`        | TEXT        | Receiving player name                              |
| `pass_recipient_id`     | REAL        | Receiving player ID                                |
| `pass_assisted_shot_id` | TEXT        | Linked shot ID if assist                           |
| `pass_goal_assist`      | INTEGER     | Goal assist flag (0/1)                             |
| `pass_shot_assist`      | INTEGER     | Shot assist flag (0/1)                             |
| `pass_cross`            | INTEGER     | Cross flag (0/1)                                   |
| `pass_switch`           | INTEGER     | Switch play flag (0/1)                             |
| `pass_through_ball`     | INTEGER     | Through ball flag (0/1)                            |
| `pass_aerial_won`       | INTEGER     | Aerial duel won (0/1)                              |
| `pass_deflected`        | INTEGER     | Deflection flag (0/1)                              |
| `pass_inswinging`       | INTEGER     | Inswinging flag (0/1)                              |
| `pass_outswinging`      | INTEGER     | Outswinging flag (0/1)                             |
| `pass_no_touch`         | INTEGER     | No touch flag (0/1)                                |
| `pass_cut_back`         | INTEGER     | Cut back flag (0/1)                                |
| `pass_straight`         | INTEGER     | Straight pass flag (0/1)                           |
| `pass_miscommunication` | INTEGER     | Miscommunication flag (0/1)                        |

##### Shot-Specific Fields

| Column                  | Type        | Description                              |
| ----------------------- | ----------- | ---------------------------------------- |
| `shot_end_location`     | TEXT (JSON) | Shot target [x, y, z]                    |
| `shot_statsbomb_xg`     | REAL        | Expected goals value (0.0-1.0)           |
| `shot_outcome`          | TEXT        | Shot result (Goal, Saved, Blocked, etc.) |
| `shot_technique`        | TEXT        | Shot technique (Normal, Volley, etc.)    |
| `shot_body_part`        | TEXT        | Body part used                           |
| `shot_type`             | TEXT        | Shot type (Open Play, Penalty, etc.)     |
| `shot_first_time`       | INTEGER     | First time shot flag (0/1)               |
| `shot_deflected`        | INTEGER     | Deflection flag (0/1)                    |
| `shot_aerial_won`       | INTEGER     | Aerial duel won (0/1)                    |
| `shot_follows_dribble`  | INTEGER     | Follows dribble flag (0/1)               |
| `shot_one_on_one`       | INTEGER     | One-on-one situation (0/1)               |
| `shot_open_goal`        | INTEGER     | Open goal flag (0/1)                     |
| `shot_redirect`         | INTEGER     | Redirect flag (0/1)                      |
| `shot_saved_off_target` | INTEGER     | Saved off target (0/1)                   |
| `shot_saved_to_post`    | INTEGER     | Saved to post (0/1)                      |
| `shot_key_pass_id`      | TEXT        | Key pass ID                              |
| `shot_freeze_frame`     | TEXT (JSON) | Player positions at shot moment          |

##### Goalkeeper-Specific Fields

| Column                             | Type        | Description                 |
| ---------------------------------- | ----------- | --------------------------- |
| `goalkeeper_end_location`          | TEXT (JSON) | Goalkeeper position [x, y]  |
| `goalkeeper_outcome`               | TEXT        | Save result                 |
| `goalkeeper_technique`             | TEXT        | Save technique              |
| `goalkeeper_body_part`             | TEXT        | Body part used              |
| `goalkeeper_type`                  | TEXT        | Action type                 |
| `goalkeeper_position`              | TEXT        | Goalkeeper position         |
| `goalkeeper_punched_out`           | INTEGER     | Punch flag (0/1)            |
| `goalkeeper_lost_in_play`          | INTEGER     | Lost in play (0/1)          |
| `goalkeeper_penalty_saved_to_post` | INTEGER     | Penalty saved to post (0/1) |
| `goalkeeper_shot_saved_off_target` | INTEGER     | Shot saved off target (0/1) |
| `goalkeeper_shot_saved_to_post`    | INTEGER     | Shot saved to post (0/1)    |
| `goalkeeper_success_in_play`       | INTEGER     | Success in play (0/1)       |

##### Other Event Fields

| Column                           | Type         | Description                  |
| -------------------------------- | ------------ | ---------------------------- |
| `under_pressure`                 | INTEGER      | Under pressure flag (0/1)    |
| `off_camera`                     | INTEGER      | Event off camera (0/1)       |
| `out_play`                       | INTEGER      | Ball out of play (0/1)       |
| `fifty_fifty`                    | INTEGER      | 50/50 duel (0/1)             |
| `duel_type`                      | TEXT         | Duel type (Aerial, Tackle)   |
| `duel_outcome`                   | TEXT         | Duel result                  |
| `tactics`                        | TEXT (JSON)  | Formation/tactical data      |
| `related_events`                 | TEXT (JSON)  | Related event IDs            |
| `carry_end_location`             | TEXT (JSON)  | Carry destination [x, y]     |
| `ball_recovery_offensive`        | INTEGER      | Offensive recovery (0/1)     |
| `ball_recovery_recovery_failure` | INTEGER      | Recovery failure (0/1)       |
| `block_deflection`               | INTEGER      | Block deflection (0/1)       |
| `block_offensive`                | INTEGER      | Offensive block (0/1)        |
| `block_save_block`               | INTEGER      | Save block (0/1)             |
| `clearance_*`                    | INTEGER/TEXT | Various clearance attributes |
| `foul_*`                         | INTEGER/TEXT | Various foul attributes      |
| `substitution_*`                 | INTEGER/TEXT | Substitution details         |

**Row Count**: 192,692 rows (~3,800 events per match)
**Total Columns**: 115

#### 3. `lineups` - Player Lineups

**Purpose**: Starting lineups and squad information for each match

| Column            | Type    | Constraints               | Description                   | Example                                                      |
| ----------------- | ------- | ------------------------- | ----------------------------- | ------------------------------------------------------------ |
| `id`              | INTEGER | PRIMARY KEY AUTOINCREMENT | Auto-generated ID             | 1                                                            |
| `match_id`        | INTEGER | FOREIGN KEY               | References matches(match_id)  | 3788741                                                      |
| `player_id`       | INTEGER |                           | Unique player identifier      | 5503                                                         |
| `player_name`     | TEXT    |                           | Player full name              | "Granit Xhaka"                                               |
| `player_nickname` | TEXT    |                           | Player nickname               | "Granit"                                                     |
| `jersey_number`   | INTEGER |                           | Jersey number                 | 10                                                           |
| `country`         | TEXT    |                           | Player nationality            | "Switzerland"                                                |
| `cards`           | TEXT    | JSON                      | Cards received during match   | `[]` or `[{"time": 45, "card": "Yellow"}]`                   |
| `positions`       | TEXT    | JSON                      | Positions played during match | `[{"position": "Center Midfield", "from": "1", "to": "90"}]` |
| `team_name`       | TEXT    |                           | Team name                     | "Switzerland"                                                |

**Row Count**: 2,345 rows (~46 players per match)

#### 4. `frames_360` - 360° Tracking Data

**Purpose**: Player tracking data with positions and visible areas

| Column         | Type    | Constraints | Description                       | Example                    |
| -------------- | ------- | ----------- | --------------------------------- | -------------------------- |
| `id`           | TEXT    |             | Frame/event identifier            | "abc123-def456"            |
| `match_id`     | INTEGER | FOREIGN KEY | References matches(match_id)      | 3788741                    |
| `visible_area` | TEXT    | JSON        | Camera visible area polygon       | `[0.0, 27.76, 19.23, ...]` |
| `teammate`     | INTEGER |             | Player is teammate of actor (0/1) | 1                          |
| `actor`        | INTEGER |             | Player is the event actor (0/1)   | 0                          |
| `keeper`       | INTEGER |             | Player is goalkeeper (0/1)        | 0                          |
| `location`     | TEXT    | JSON        | Player position [x, y]            | `[43.65, 31.99]`           |

**Row Count**: 2,538,580 rows (~50,000 frames per match)

### Views

#### 1. `goals` - Goal Analysis View

**Purpose**: Simplified view of all goals with key details

```sql
CREATE VIEW goals AS
SELECT
    e.match_id,
    m.home_team,
    m.away_team,
    e.team,
    e.player,
    e.minute,
    e.second,
    e.shot_statsbomb_xg,
    json_extract(e.location, '$[0]') as x_coord,
    json_extract(e.location, '$[1]') as y_coord
FROM events e
JOIN matches m ON e.match_id = m.match_id
WHERE e.shot_outcome = 'Goal'
ORDER BY e.match_id, e.minute, e.second
```

**Columns**: match_id, home_team, away_team, team, player, minute, second, shot_statsbomb_xg, x_coord, y_coord
**Row Count**: 155 goals

#### 2. `passes` - Pass Analysis View

**Purpose**: Simplified view of all passes with coordinate extraction

```sql
CREATE VIEW passes AS
SELECT
    e.match_id,
    e.team,
    e.player,
    e.minute,
    e.pass_length,
    e.pass_angle,
    e.pass_outcome,
    json_extract(e.location, '$[0]') as start_x,
    json_extract(e.location, '$[1]') as start_y,
    json_extract(e.pass_end_location, '$[0]') as end_x,
    json_extract(e.pass_end_location, '$[1]') as end_y
FROM events e
WHERE e.type = 'Pass'
AND e.location IS NOT NULL
AND e.pass_end_location IS NOT NULL
```

**Columns**: match_id, team, player, minute, pass_length, pass_angle, pass_outcome, start_x, start_y, end_x, end_y
**Row Count**: 54,819 passes

### Indexes

Performance indexes for common queries:

```sql
CREATE INDEX idx_events_match_id ON events (match_id);
CREATE INDEX idx_events_type ON events (type);
CREATE INDEX idx_events_player_id ON events (player_id);
CREATE INDEX idx_events_team_id ON events (team_id);
CREATE INDEX idx_events_minute ON events (minute);
CREATE INDEX idx_lineups_match_id ON lineups (match_id);
CREATE INDEX idx_lineups_player_id ON lineups (player_id);
CREATE INDEX idx_frames_match_id ON frames_360 (match_id);
CREATE INDEX idx_frames_id ON frames_360 (id);
```

## Data Types and Conventions

### Coordinate System

- **Field Dimensions**: 120 yards (width) × 80 yards (height)
- **Origin**: Bottom-left corner (0, 0)
- **Goals**: Located at x=0 (left goal) and x=120 (right goal)
- **Coordinate Range**: x: 0.1-120.0, y: 0.1-80.0 (actual data range)
- **Format**: JSON arrays `[x, y]` for 2D, `[x, y, z]` for 3D (shots)

### JSON Fields

All JSON fields are stored as TEXT and can be parsed using SQLite's JSON functions:

#### Location Fields

```sql
-- Extract x coordinate
json_extract(location, '$[0]')

-- Extract y coordinate
json_extract(location, '$[1]')

-- Extract z coordinate (shots)
json_extract(shot_end_location, '$[2]')
```

#### Complex JSON Structures

```sql
-- Extract formation from tactics
json_extract(tactics, '$.formation')

-- Extract visible area points
json_extract(visible_area, '$[0]')  -- First point
```

### Boolean Fields

All boolean values are stored as INTEGER:

- `0` = False/No
- `1` = True/Yes
- `NULL` = Not applicable/Unknown

### Event Types

Common event types in the database:

- **Pass** (54,819 events) - All types of passes
- **Ball Receipt\*** (52,721 events) - Ball receptions
- **Carry** (43,801 events) - Ball carries/dribbles
- **Pressure** (15,958 events) - Defensive pressure
- **Ball Recovery** (4,445 events) - Ball recoveries
- **Duel** (3,355 events) - 1v1 duels
- **Clearance** (2,283 events) - Defensive clearances
- **Block** (1,770 events) - Shot/pass blocks
- **Goal Keeper** (1,514 events) - Goalkeeper actions
- **Dribble** (1,476 events) - Dribbling actions
- **Interception** (1,460 events) - Ball interceptions
- **Foul Committed** (1,394 events) - Fouls committed
- **Shot** (1,289 events) - All shots (including goals)

## Query Examples

### Basic Queries

#### Get match information

```sql
SELECT home_team, away_team, home_score, away_score, match_date
FROM matches
ORDER BY match_date;
```

#### Count events by type

```sql
SELECT type, COUNT(*) as count
FROM events
GROUP BY type
ORDER BY count DESC;
```

#### Find all goals

```sql
SELECT player, team, minute, shot_statsbomb_xg
FROM goals
ORDER BY shot_statsbomb_xg DESC;
```

### Advanced Analytics

#### Top goal scorers

```sql
SELECT
    player,
    team,
    COUNT(*) as goals,
    AVG(shot_statsbomb_xg) as avg_xg
FROM goals
GROUP BY player, team
ORDER BY goals DESC, avg_xg DESC;
```

#### Pass completion rates by team

```sql
SELECT
    team,
    COUNT(*) as total_passes,
    SUM(CASE WHEN pass_outcome IS NULL THEN 1 ELSE 0 END) as completed_passes,
    ROUND(SUM(CASE WHEN pass_outcome IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as completion_rate
FROM passes
GROUP BY team
ORDER BY completion_rate DESC;
```

#### Heat map data for a player

```sql
SELECT
    json_extract(location, '$[0]') as x,
    json_extract(location, '$[1]') as y,
    COUNT(*) as event_count
FROM events
WHERE player = 'Granit Xhaka'
AND location IS NOT NULL
GROUP BY x, y
ORDER BY event_count DESC;
```

#### Expected goals vs actual goals by team

```sql
WITH team_stats AS (
    SELECT
        m.home_team as team,
        SUM(m.home_score) as actual_goals,
        ROUND(SUM(CASE WHEN e.shot_outcome = 'Goal' AND e.team = m.home_team
                       THEN e.shot_statsbomb_xg ELSE 0 END), 2) as expected_goals
    FROM matches m
    LEFT JOIN events e ON m.match_id = e.match_id AND e.type = 'Shot'
    GROUP BY m.home_team

    UNION ALL

    SELECT
        m.away_team as team,
        SUM(m.away_score) as actual_goals,
        ROUND(SUM(CASE WHEN e.shot_outcome = 'Goal' AND e.team = m.away_team
                       THEN e.shot_statsbomb_xg ELSE 0 END), 2) as expected_goals
    FROM matches m
    LEFT JOIN events e ON m.match_id = e.match_id AND e.type = 'Shot'
    GROUP BY m.away_team
)
SELECT team, SUM(actual_goals) as total_goals, SUM(expected_goals) as total_xg
FROM team_stats
GROUP BY team
ORDER BY total_xg DESC;
```

### Performance Considerations

#### Query Optimization

- Use indexes on frequently queried columns (match_id, type, player_id)
- Filter early in WHERE clauses
- Use JSON functions efficiently
- Consider query result caching for complex analytics

#### Memory Usage

- SQLite loads entire database into memory if possible (~845 MB RAM)
- Large result sets may require chunking (especially frames_360 queries)
- JSON extraction can be memory intensive for large datasets
- Consider using LIMIT clauses for exploration queries

## Data Quality and Validation

### Data Integrity Checks

```sql
-- Check for orphaned events
SELECT COUNT(*) FROM events e
LEFT JOIN matches m ON e.match_id = m.match_id
WHERE m.match_id IS NULL;

-- Validate coordinate ranges
SELECT COUNT(*) FROM events
WHERE json_extract(location, '$[0]') < 0
   OR json_extract(location, '$[0]') > 120
   OR json_extract(location, '$[1]') < 0
   OR json_extract(location, '$[1]') > 80;

-- Check xG value ranges
SELECT MIN(shot_statsbomb_xg), MAX(shot_statsbomb_xg)
FROM events
WHERE shot_statsbomb_xg IS NOT NULL;
```

### Known Data Characteristics

- All matches have complete event data
- 360° tracking data available for all matches
- xG values range from 0.005 to 0.963 (average ~0.127)
- Match dates: 2021-06-11 to 2021-07-11 (31 days total)
- Maximum goals in a match: 8 total (Croatia 3-5 Spain)
- Average events per match: ~3,779
- Average tracking frames per match: ~49,776
- No missing or corrupted data detected

## Usage Guidelines

### Best Practices

1. **Always use indexes**: Filter on indexed columns (match_id, type, player_id) when possible
2. **JSON efficiency**: Extract JSON values once and store in CTEs for complex queries
3. **Coordinate validation**: Check coordinate bounds (0.1-120.0, 0.1-80.0) for spatial analysis
4. **Event relationships**: Use related_events for multi-event sequences
5. **Time analysis**: Combine minute and second for precise timing
6. **Memory management**: Use LIMIT clauses for large queries, especially on frames_360
7. **Data validation**: Always check for NULL values in optional fields

### Common Pitfalls

1. **Boolean interpretation**: Remember 0/1 encoding for boolean fields
2. **NULL handling**: Many optional fields are NULL, not empty strings
3. **Team consistency**: Team names are consistent but check for exact matches
4. **Coordinate system**: StatsBomb coordinates differ from other providers
5. **Event ordering**: Use index_num for proper event sequence

## License and Attribution

- **Data Source**: StatsBomb Open Data
- **License**: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial)
- **Usage**: Non-commercial use only, attribution required
- **Citation**: "StatsBomb Open Data, UEFA Euro 2020"

## Version History

- **v1.0**: Initial SQLite migration from StatsBomb parquet files
- **Schema Version**: Dynamic schema supporting all 115 event columns
- **Migration Date**: 2025-09-03
- **Total Records**: 2,733,668
- **Migration Duration**: ~58 seconds
- **Migration Tool**: `migrate_to_sqlite.py`
