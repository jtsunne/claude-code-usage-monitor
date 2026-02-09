"""Comprehensive tests for team plans (team_premium, team_standard).

Covers PlanType enum, PlanConfig, Plans class methods, SessionBlock sonnet tracking,
is_sonnet_model helper, SessionAnalyzer with 168h duration, weekly time display format,
team plan UI rendering, and CLI override limits.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from claude_monitor.core.models import (
    SessionBlock,
    TokenCounts,
    UsageEntry,
    is_sonnet_model,
)
from claude_monitor.core.plans import (
    PLAN_LIMITS,
    Plans,
    PlanType,
)
from claude_monitor.data.analyzer import SessionAnalyzer
from claude_monitor.ui.session_display import SessionDisplayComponent

# ---------------------------------------------------------------------------
# PlanType enum
# ---------------------------------------------------------------------------


class TestPlanTypeTeamValues:
    """PlanType enum includes team plan members."""

    def test_team_premium_exists(self) -> None:
        assert PlanType.TEAM_PREMIUM.value == "team_premium"

    def test_team_standard_exists(self) -> None:
        assert PlanType.TEAM_STANDARD.value == "team_standard"


class TestPlanTypeFromString:
    """PlanType.from_string() handles team plan strings."""

    def test_from_string_team_premium(self) -> None:
        assert PlanType.from_string("team_premium") is PlanType.TEAM_PREMIUM

    def test_from_string_team_standard(self) -> None:
        assert PlanType.from_string("team_standard") is PlanType.TEAM_STANDARD

    def test_from_string_team_premium_uppercase(self) -> None:
        assert PlanType.from_string("TEAM_PREMIUM") is PlanType.TEAM_PREMIUM

    def test_from_string_team_standard_mixed_case(self) -> None:
        assert PlanType.from_string("Team_Standard") is PlanType.TEAM_STANDARD

    def test_from_string_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown plan type"):
            PlanType.from_string("nonexistent_plan")


# ---------------------------------------------------------------------------
# PlanConfig for team plans
# ---------------------------------------------------------------------------


class TestPlanConfigTeamPremium:
    """PlanConfig values for Team Premium."""

    def setup_method(self) -> None:
        self.cfg = Plans.get_plan(PlanType.TEAM_PREMIUM)

    def test_token_limit(self) -> None:
        assert self.cfg.token_limit == 118_750

    def test_sonnet_token_limit(self) -> None:
        assert self.cfg.sonnet_token_limit == 593_750

    def test_session_duration_hours(self) -> None:
        assert self.cfg.session_duration_hours == 168

    def test_is_weekly(self) -> None:
        assert self.cfg.is_weekly is True

    def test_price_monthly(self) -> None:
        assert self.cfg.price_monthly == 150.0

    def test_cost_limit(self) -> None:
        assert self.cfg.cost_limit == 112.5

    def test_message_limit(self) -> None:
        assert self.cfg.message_limit == 1_562

    def test_display_name(self) -> None:
        assert self.cfg.display_name == "Team Premium"


class TestPlanConfigTeamStandard:
    """PlanConfig values for Team Standard."""

    def setup_method(self) -> None:
        self.cfg = Plans.get_plan(PlanType.TEAM_STANDARD)

    def test_token_limit(self) -> None:
        assert self.cfg.token_limit == 23_750

    def test_sonnet_token_limit_is_none(self) -> None:
        assert self.cfg.sonnet_token_limit is None

    def test_session_duration_hours(self) -> None:
        assert self.cfg.session_duration_hours == 168

    def test_is_weekly(self) -> None:
        assert self.cfg.is_weekly is True

    def test_price_monthly(self) -> None:
        assert self.cfg.price_monthly == 30.0

    def test_cost_limit(self) -> None:
        assert self.cfg.cost_limit == 22.5

    def test_message_limit(self) -> None:
        assert self.cfg.message_limit == 312

    def test_display_name(self) -> None:
        assert self.cfg.display_name == "Team Standard"


# ---------------------------------------------------------------------------
# Plans helper methods
# ---------------------------------------------------------------------------


class TestPlansIsTeamPlan:
    """Plans.is_team_plan() behaviour."""

    def test_team_premium_returns_true(self) -> None:
        assert Plans.is_team_plan("team_premium") is True

    def test_team_standard_returns_true(self) -> None:
        assert Plans.is_team_plan("team_standard") is True

    def test_pro_returns_false(self) -> None:
        assert Plans.is_team_plan("pro") is False

    def test_max5_returns_false(self) -> None:
        assert Plans.is_team_plan("max5") is False

    def test_custom_returns_false(self) -> None:
        assert Plans.is_team_plan("custom") is False

    def test_unknown_returns_false(self) -> None:
        assert Plans.is_team_plan("unknown_plan") is False


class TestPlansGetSessionDurationHours:
    """Plans.get_session_duration_hours() behaviour."""

    def test_team_premium_returns_168(self) -> None:
        assert Plans.get_session_duration_hours("team_premium") == 168

    def test_team_standard_returns_168(self) -> None:
        assert Plans.get_session_duration_hours("team_standard") == 168

    def test_pro_returns_5(self) -> None:
        assert Plans.get_session_duration_hours("pro") == 5

    def test_max5_returns_5(self) -> None:
        assert Plans.get_session_duration_hours("max5") == 5

    def test_max20_returns_5(self) -> None:
        assert Plans.get_session_duration_hours("max20") == 5

    def test_custom_returns_5(self) -> None:
        assert Plans.get_session_duration_hours("custom") == 5

    def test_unknown_returns_5(self) -> None:
        assert Plans.get_session_duration_hours("invalid") == 5


class TestPlansGetSonnetTokenLimit:
    """Plans.get_sonnet_token_limit() behaviour."""

    def test_team_premium_has_sonnet_limit(self) -> None:
        assert Plans.get_sonnet_token_limit("team_premium") == 593_750

    def test_team_standard_has_no_sonnet_limit(self) -> None:
        assert Plans.get_sonnet_token_limit("team_standard") is None

    def test_pro_has_no_sonnet_limit(self) -> None:
        assert Plans.get_sonnet_token_limit("pro") is None

    def test_unknown_returns_none(self) -> None:
        assert Plans.get_sonnet_token_limit("nonexistent") is None


class TestPlansGetWeeklyPlans:
    """Plans.get_weekly_plans() returns team plans only."""

    def test_contains_team_premium(self) -> None:
        weekly = Plans.get_weekly_plans()
        assert PlanType.TEAM_PREMIUM in weekly

    def test_contains_team_standard(self) -> None:
        weekly = Plans.get_weekly_plans()
        assert PlanType.TEAM_STANDARD in weekly

    def test_does_not_contain_pro(self) -> None:
        weekly = Plans.get_weekly_plans()
        assert PlanType.PRO not in weekly

    def test_returns_only_two_plans(self) -> None:
        weekly = Plans.get_weekly_plans()
        assert len(weekly) == 2


# ---------------------------------------------------------------------------
# PlanConfig.is_team_plan property
# ---------------------------------------------------------------------------


class TestPlanConfigIsTeamPlan:
    """PlanConfig.is_team_plan property."""

    def test_team_premium_config_is_team(self) -> None:
        cfg = Plans.get_plan(PlanType.TEAM_PREMIUM)
        assert cfg.is_team_plan is True

    def test_team_standard_config_is_team(self) -> None:
        cfg = Plans.get_plan(PlanType.TEAM_STANDARD)
        assert cfg.is_team_plan is True

    def test_pro_config_is_not_team(self) -> None:
        cfg = Plans.get_plan(PlanType.PRO)
        assert cfg.is_team_plan is False


# ---------------------------------------------------------------------------
# SessionBlock.sonnet_token_counts accumulation
# ---------------------------------------------------------------------------


class TestSessionBlockSonnetTokenCounts:
    """SessionBlock.sonnet_token_counts and sonnet_total_tokens."""

    def test_default_sonnet_token_counts_are_zero(self) -> None:
        block = SessionBlock(
            id="test",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc),
        )
        assert block.sonnet_total_tokens == 0

    def test_sonnet_token_counts_accumulate(self) -> None:
        block = SessionBlock(
            id="test",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc),
            sonnet_token_counts=TokenCounts(
                input_tokens=100,
                output_tokens=200,
                cache_creation_tokens=50,
                cache_read_tokens=25,
            ),
        )
        assert block.sonnet_total_tokens == 375

    def test_sonnet_total_tokens_property(self) -> None:
        counts = TokenCounts(input_tokens=10, output_tokens=20)
        block = SessionBlock(
            id="test",
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 5, 0, tzinfo=timezone.utc),
            sonnet_token_counts=counts,
        )
        assert block.sonnet_total_tokens == counts.total_tokens


# ---------------------------------------------------------------------------
# is_sonnet_model() helper
# ---------------------------------------------------------------------------


class TestIsSonnetModel:
    """is_sonnet_model() correctly identifies Sonnet models."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet",
            "claude-3-sonnet",
            "Sonnet",
            "SONNET",
            "claude-sonnet-4-20250514",
        ],
    )
    def test_sonnet_models_return_true(self, model: str) -> None:
        assert is_sonnet_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-3-haiku",
            "claude-haiku-4-5-20251001",
            "unknown",
            "",
        ],
    )
    def test_non_sonnet_models_return_false(self, model: str) -> None:
        assert is_sonnet_model(model) is False


# ---------------------------------------------------------------------------
# SessionAnalyzer with 168h duration
# ---------------------------------------------------------------------------


class TestSessionAnalyzer168h:
    """SessionAnalyzer with session_duration_hours=168 creates weekly blocks."""

    def test_init_168h(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        assert analyzer.session_duration_hours == 168
        assert analyzer.session_duration == timedelta(hours=168)

    def test_block_end_time_is_168h_from_start(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        entry = UsageEntry(
            timestamp=datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc),
            input_tokens=100,
            output_tokens=50,
            model="claude-3-haiku",
        )
        block = analyzer._create_new_block(entry)
        expected_end = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc) + timedelta(
            hours=168
        )
        assert block.end_time == expected_end

    def test_entries_within_168h_stay_in_one_block(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        entries: List[UsageEntry] = [
            UsageEntry(
                timestamp=base,
                input_tokens=100,
                output_tokens=50,
                model="claude-3-haiku",
            ),
            UsageEntry(
                timestamp=base + timedelta(hours=24),
                input_tokens=200,
                output_tokens=100,
                model="claude-sonnet-4-5-20250929",
            ),
            UsageEntry(
                timestamp=base + timedelta(hours=100),
                input_tokens=300,
                output_tokens=150,
                model="claude-opus-4-6",
            ),
        ]
        blocks = analyzer.transform_to_blocks(entries)
        # All entries within 168h window should be in one block
        non_gap = [b for b in blocks if not b.is_gap]
        assert len(non_gap) == 1
        assert len(non_gap[0].entries) == 3

    def test_entries_beyond_168h_create_new_block(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        base = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        entries: List[UsageEntry] = [
            UsageEntry(
                timestamp=base,
                input_tokens=100,
                output_tokens=50,
                model="claude-3-haiku",
            ),
            UsageEntry(
                timestamp=base + timedelta(hours=200),
                input_tokens=200,
                output_tokens=100,
                model="claude-3-haiku",
            ),
        ]
        blocks = analyzer.transform_to_blocks(entries)
        non_gap = [b for b in blocks if not b.is_gap]
        assert len(non_gap) == 2


# ---------------------------------------------------------------------------
# Sonnet token accumulation in _add_entry_to_block
# ---------------------------------------------------------------------------


class TestAddEntryToBlockSonnetAccumulation:
    """_add_entry_to_block correctly accumulates Sonnet tokens."""

    def _make_block(self) -> SessionBlock:
        return SessionBlock(
            id="test",
            start_time=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 8, 12, 0, tzinfo=timezone.utc),
            token_counts=TokenCounts(),
            sonnet_token_counts=TokenCounts(),
        )

    def test_sonnet_entry_accumulates_sonnet_tokens(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        block = self._make_block()
        entry = UsageEntry(
            timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            input_tokens=500,
            output_tokens=250,
            cache_creation_tokens=100,
            cache_read_tokens=50,
            model="claude-sonnet-4-5-20250929",
        )
        analyzer._add_entry_to_block(block, entry)

        assert block.sonnet_token_counts.input_tokens == 500
        assert block.sonnet_token_counts.output_tokens == 250
        assert block.sonnet_token_counts.cache_creation_tokens == 100
        assert block.sonnet_token_counts.cache_read_tokens == 50
        assert block.sonnet_total_tokens == 900

    def test_non_sonnet_entry_does_not_accumulate_sonnet_tokens(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        block = self._make_block()
        entry = UsageEntry(
            timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            input_tokens=500,
            output_tokens=250,
            model="claude-opus-4-6",
        )
        analyzer._add_entry_to_block(block, entry)

        assert block.sonnet_total_tokens == 0
        # But all-model counts should still accumulate
        assert block.token_counts.input_tokens == 500
        assert block.token_counts.output_tokens == 250

    def test_mixed_models_accumulate_correctly(self) -> None:
        analyzer = SessionAnalyzer(session_duration_hours=168)
        block = self._make_block()

        sonnet_entry = UsageEntry(
            timestamp=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            input_tokens=100,
            output_tokens=50,
            model="claude-sonnet-4-5-20250929",
        )
        opus_entry = UsageEntry(
            timestamp=datetime(2024, 1, 1, 14, 0, tzinfo=timezone.utc),
            input_tokens=200,
            output_tokens=100,
            model="claude-opus-4-6",
        )
        sonnet_entry_2 = UsageEntry(
            timestamp=datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc),
            input_tokens=300,
            output_tokens=150,
            model="claude-3-5-sonnet",
        )

        analyzer._add_entry_to_block(block, sonnet_entry)
        analyzer._add_entry_to_block(block, opus_entry)
        analyzer._add_entry_to_block(block, sonnet_entry_2)

        # All models total
        assert block.token_counts.input_tokens == 600
        assert block.token_counts.output_tokens == 300

        # Sonnet-only total
        assert block.sonnet_token_counts.input_tokens == 400
        assert block.sonnet_token_counts.output_tokens == 200
        assert block.sonnet_total_tokens == 600


# ---------------------------------------------------------------------------
# Weekly time display format (Xd Xh Xm)
# ---------------------------------------------------------------------------


class TestWeeklyTimeDisplayFormat:
    """Team plan time display uses 'Xd Xh Xm' format."""

    def test_team_premium_screen_contains_weekly_time_format(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_premium",
            timezone="UTC",
            tokens_used=10_000,
            token_limit=118_750,
            usage_percentage=8.4,
            tokens_left=108_750,
            elapsed_session_minutes=60 * 24,  # 1 day
            total_session_minutes=168 * 60,  # 7 days
            burn_rate=10.0,
            session_cost=5.0,
            per_model_stats={},
            sent_messages=50,
            entries=[],
            predicted_end_str="Jan 08 10:00",
            reset_time_str="Jan 08 12:00",
            current_time_str="Jan 01 12:00:00",
        )
        joined = "\n".join(lines)
        # Should contain the "Xd Xh Xm" format
        assert "d " in joined and "h " in joined and "m" in joined

    def test_non_team_plan_uses_hours_minutes(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="pro",
            timezone="UTC",
            tokens_used=5_000,
            token_limit=19_000,
            usage_percentage=26.3,
            tokens_left=14_000,
            elapsed_session_minutes=60,
            total_session_minutes=300,
            burn_rate=10.0,
            session_cost=1.0,
            per_model_stats={},
            sent_messages=20,
            entries=[],
            predicted_end_str="Jan 01 16:00",
            reset_time_str="Jan 01 17:00",
            current_time_str="Jan 01 13:00:00",
        )
        joined = "\n".join(lines)
        # Standard plans use "Xh Xm" without days
        assert "Time to Reset" in joined


# ---------------------------------------------------------------------------
# Team premium dual progress bar rendering
# ---------------------------------------------------------------------------


class TestTeamPremiumDualProgressBar:
    """Team Premium shows both All Models and Sonnet Only progress bars."""

    def test_all_models_and_sonnet_bars_present(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_premium",
            timezone="UTC",
            tokens_used=50_000,
            token_limit=118_750,
            usage_percentage=42.1,
            tokens_left=68_750,
            elapsed_session_minutes=1440,
            total_session_minutes=168 * 60,
            burn_rate=5.0,
            session_cost=20.0,
            per_model_stats={},
            sent_messages=100,
            entries=[],
            predicted_end_str="Jan 08 10:00",
            reset_time_str="Jan 08 12:00",
            current_time_str="Jan 02 12:00:00",
            sonnet_tokens_used=30_000,
            sonnet_token_limit=593_750,
        )
        joined = "\n".join(lines)
        assert "All Models:" in joined
        assert "Sonnet Only:" in joined

    def test_weekly_usage_limits_header(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_premium",
            timezone="UTC",
            tokens_used=0,
            token_limit=118_750,
            usage_percentage=0,
            tokens_left=118_750,
            elapsed_session_minutes=0,
            total_session_minutes=168 * 60,
            burn_rate=0.0,
            session_cost=0.0,
            per_model_stats={},
            sent_messages=0,
            entries=[],
            predicted_end_str="--",
            reset_time_str="--",
            current_time_str="--",
        )
        joined = "\n".join(lines)
        assert "Weekly Usage Limits" in joined
        assert "Team Premium" in joined
        assert "$150" in joined

    def test_cost_and_messages_bars_present(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_premium",
            timezone="UTC",
            tokens_used=10_000,
            token_limit=118_750,
            usage_percentage=8.4,
            tokens_left=108_750,
            elapsed_session_minutes=60,
            total_session_minutes=168 * 60,
            burn_rate=5.0,
            session_cost=10.0,
            per_model_stats={},
            sent_messages=50,
            entries=[],
            predicted_end_str="--",
            reset_time_str="--",
            current_time_str="--",
        )
        joined = "\n".join(lines)
        assert "Cost Usage:" in joined
        assert "Messages Usage:" in joined


# ---------------------------------------------------------------------------
# Team standard single progress bar rendering
# ---------------------------------------------------------------------------


class TestTeamStandardSingleProgressBar:
    """Team Standard shows All Models bar but NO Sonnet bar."""

    def test_all_models_bar_present(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_standard",
            timezone="UTC",
            tokens_used=5_000,
            token_limit=23_750,
            usage_percentage=21.1,
            tokens_left=18_750,
            elapsed_session_minutes=720,
            total_session_minutes=168 * 60,
            burn_rate=3.0,
            session_cost=3.0,
            per_model_stats={},
            sent_messages=30,
            entries=[],
            predicted_end_str="--",
            reset_time_str="--",
            current_time_str="--",
        )
        joined = "\n".join(lines)
        assert "All Models:" in joined

    def test_no_sonnet_bar_for_team_standard(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_standard",
            timezone="UTC",
            tokens_used=5_000,
            token_limit=23_750,
            usage_percentage=21.1,
            tokens_left=18_750,
            elapsed_session_minutes=720,
            total_session_minutes=168 * 60,
            burn_rate=3.0,
            session_cost=3.0,
            per_model_stats={},
            sent_messages=30,
            entries=[],
            predicted_end_str="--",
            reset_time_str="--",
            current_time_str="--",
        )
        joined = "\n".join(lines)
        assert "Sonnet Only:" not in joined

    def test_team_standard_header(self) -> None:
        component = SessionDisplayComponent()
        lines = component.format_active_session_screen(
            plan="team_standard",
            timezone="UTC",
            tokens_used=0,
            token_limit=23_750,
            usage_percentage=0,
            tokens_left=23_750,
            elapsed_session_minutes=0,
            total_session_minutes=168 * 60,
            burn_rate=0.0,
            session_cost=0.0,
            per_model_stats={},
            sent_messages=0,
            entries=[],
            predicted_end_str="--",
            reset_time_str="--",
            current_time_str="--",
        )
        joined = "\n".join(lines)
        assert "Weekly Usage Limits" in joined
        assert "Team Standard" in joined
        assert "$30" in joined


# ---------------------------------------------------------------------------
# CLI override limits (weekly_all_models_limit, weekly_sonnet_limit)
# ---------------------------------------------------------------------------


class TestCLIOverrideLimits:
    """DisplayController._process_active_session_data applies CLI overrides."""

    def _make_args(self, **overrides: Any) -> Mock:
        args = Mock()
        args.plan = overrides.get("plan", "team_premium")
        args.timezone = overrides.get("timezone", "UTC")
        args.custom_limit_tokens = overrides.get("custom_limit_tokens", None)
        args.weekly_all_models_limit = overrides.get("weekly_all_models_limit", None)
        args.weekly_sonnet_limit = overrides.get("weekly_sonnet_limit", None)
        return args

    def _make_active_block(self, **overrides: Any) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "id": "test-block",
            "isActive": True,
            "totalTokens": overrides.get("totalTokens", 50_000),
            "costUSD": overrides.get("costUSD", 10.0),
            "startTime": overrides.get(
                "startTime", (now - timedelta(hours=24)).isoformat()
            ),
            "endTime": overrides.get(
                "endTime", (now + timedelta(hours=144)).isoformat()
            ),
            "perModelStats": overrides.get("perModelStats", {}),
            "sentMessagesCount": overrides.get("sentMessagesCount", 50),
            "entries": overrides.get("entries", []),
            "sonnetTotalTokens": overrides.get("sonnetTotalTokens", 20_000),
        }

    def test_weekly_all_models_override_applied(self) -> None:
        from claude_monitor.ui.display_controller import DisplayController

        controller = DisplayController()
        args = self._make_args(plan="team_premium", weekly_all_models_limit=200_000)
        active_block = self._make_active_block(totalTokens=50_000)
        data = {"blocks": [active_block]}

        result = controller._process_active_session_data(
            active_block, data, args, 118_750, datetime.now(timezone.utc)
        )

        assert result["token_limit"] == 200_000
        assert result["tokens_left"] == 200_000 - 50_000

    def test_weekly_sonnet_override_applied(self) -> None:
        from claude_monitor.ui.display_controller import DisplayController

        controller = DisplayController()
        args = self._make_args(plan="team_premium", weekly_sonnet_limit=800_000)
        active_block = self._make_active_block(sonnetTotalTokens=20_000)
        data = {"blocks": [active_block]}

        result = controller._process_active_session_data(
            active_block, data, args, 118_750, datetime.now(timezone.utc)
        )

        assert result["sonnet_token_limit"] == 800_000

    def test_no_override_uses_plan_defaults(self) -> None:
        from claude_monitor.ui.display_controller import DisplayController

        controller = DisplayController()
        args = self._make_args(plan="team_premium")
        active_block = self._make_active_block()
        data = {"blocks": [active_block]}

        result = controller._process_active_session_data(
            active_block, data, args, 118_750, datetime.now(timezone.utc)
        )

        assert result["token_limit"] == 118_750
        assert result["sonnet_token_limit"] == 593_750


# ---------------------------------------------------------------------------
# PLAN_LIMITS dict coverage
# ---------------------------------------------------------------------------


class TestPlanLimitsDict:
    """PLAN_LIMITS dict has correct entries for team plans."""

    def test_team_premium_in_plan_limits(self) -> None:
        assert PlanType.TEAM_PREMIUM in PLAN_LIMITS

    def test_team_standard_in_plan_limits(self) -> None:
        assert PlanType.TEAM_STANDARD in PLAN_LIMITS

    def test_team_premium_limits_complete(self) -> None:
        data = PLAN_LIMITS[PlanType.TEAM_PREMIUM]
        required_keys = {
            "token_limit",
            "sonnet_token_limit",
            "cost_limit",
            "message_limit",
            "display_name",
            "session_duration_hours",
            "is_weekly",
            "price_monthly",
        }
        assert required_keys.issubset(data.keys())

    def test_team_standard_limits_complete(self) -> None:
        data = PLAN_LIMITS[PlanType.TEAM_STANDARD]
        required_keys = {
            "token_limit",
            "sonnet_token_limit",
            "cost_limit",
            "message_limit",
            "display_name",
            "session_duration_hours",
            "is_weekly",
            "price_monthly",
        }
        assert required_keys.issubset(data.keys())
