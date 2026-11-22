"""Regression for the firmware words observed on the deployed SNA6000."""

from homeassistant.const import ATTR_ENTITY_ID

from test_fan_controls import N1, N2, S1, S2, fan_setup, fan_state


async def test_cbaa_discovers_four_controls_and_permits_locked_verified_write(hass, fan_setup):
    dongle, setup = fan_setup
    # Holding registers 7–10 from the SNA6000: cBaa-33919A.
    dongle.hold.update({7: 0x4263, 8: 0x6161, 9: 0x9133, 10: 0x019A})
    await setup()
    assert [fan_state(hass, item).state for item in (N1, N2, S1, S2)] == [
        "100", "100", "Default", "Default"
    ]
    await hass.services.async_call(
        "select", "select_option",
        {ATTR_ENTITY_ID: fan_state(hass, S1).entity_id, "option": "High Speed"},
        blocking=True,
    )
    assert dongle.hold[178] == 0xC8C9
    assert fan_state(hass, S1).state == "High Speed"
