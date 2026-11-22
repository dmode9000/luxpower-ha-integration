"""Fan controls through Home Assistant, the Modbus client and a fake dongle."""

import asyncio

import pytest

from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.lxp_modbus.classes.modbus_client import LxpModbusApiClient
from custom_components.lxp_modbus.const import (
    CONF_BATTERY_ENTITIES,
    CONF_CONNECTION_RETRIES,
    CONF_DONGLE_SERIAL,
    CONF_ENTITY_PREFIX,
    CONF_HOST,
    CONF_INVERTER_SERIAL,
    CONF_POLL_INTERVAL,
    CONF_PORT,
    DOMAIN,
)
from fake_dongle import FakeDongle, build_frame

FAN_REG = 178
READ_FIRMWARE = (3, 7, 4)
N1 = ("number", "fan_1_max_speed")
N2 = ("number", "fan_2_max_speed")
S1 = ("select", "fan_1_speed_slope")
S2 = ("select", "fan_2_speed_slope")
CONTROLS = (N1, N2, S1, S2)


class FanDongle(FakeDongle):
    """Keep an inverter-side word independent from the coordinator snapshot."""

    def __init__(self):
        self.hold = {7: 0x6342, 8: 0x6161, 9: 0x0803, 10: 8, FAN_REG: 0xC8C8}
        self.readback_word = None
        super().__init__(value_of=lambda fn, reg: self.hold.get(reg, 0) if fn == 3 else 0)

    def write(self, req):
        super().write(req)
        function, register, value = self.requests[-1]
        if (function, register) == (6, FAN_REG):
            self.hold[register] = value if self.readback_word is None else self.readback_word


@pytest.fixture
def fan_client(monkeypatch):
    dongle = FanDongle()

    async def connect(*_args, **_kwargs):
        return dongle, dongle

    monkeypatch.setattr("asyncio.open_connection", connect)
    monkeypatch.setattr(
        "custom_components.lxp_modbus.classes.connection_manager.INITIAL_DATA_TIMEOUT", 0.01
    )
    for attr, timeout in (("READ_TIMEOUT", 0.05), ("WRITE_READ_TIMEOUT", 0.05), ("WRITE_RETRY_DELAY", 0)):
        monkeypatch.setattr(f"custom_components.lxp_modbus.classes.modbus_client.{attr}", timeout)
    client = LxpModbusApiClient(
        "192.0.2.1", 8000, dongle.dongle.decode(), dongle.serial.decode(), asyncio.Lock(),
        connection_retries=3,
    )
    return client, dongle


@pytest.mark.asyncio
async def test_overlong_fresh_response_never_composes_a_fan_write(fan_client, monkeypatch):
    client, dongle = fan_client
    original_write = dongle.write

    def write(request):
        original_write(request)
        if dongle.requests[-1] == (3, FAN_REG, 1):
            dongle._pending.clear()
            dongle.push(build_frame(dongle.dongle, dongle.serial, 3, FAN_REG, [0xB5C8, 0xC8C8]))

    monkeypatch.setattr(dongle, "write", write)
    result = await client.async_write_register(FAN_REG, 1, mask=1)

    assert (result, dongle.requests) == (False, [READ_FIRMWARE, (3, FAN_REG, 1)])
    assert dongle.hold[FAN_REG] == 0xC8C8
    assert client.get_cached_data()["hold"] == {}


@pytest.mark.parametrize("failure", ["missing", "wrong_address", "overlong", "sentinel", "wrong_bits"])
@pytest.mark.asyncio
async def test_unverified_readback_rejects_write_without_caching_echo(fan_client, monkeypatch, failure):
    client, dongle = fan_client
    original_write = dongle.write

    def write(request):
        original_write(request)
        function, register, count = dongle.requests[-1]
        if function != 3 or not any(fn == 6 for fn, *_ in dongle.requests):
            return
        dongle._pending.clear()
        if failure == "missing":
            return
        if failure == "wrong_address":
            register += 1
        if failure == "overlong":
            count += 1
        word = {"sentinel": 0xFFFF, "wrong_bits": 0xB4C8}.get(failure, 0xC8C9)
        values = [word if reg == FAN_REG else 0 for reg in range(register, register + count)]
        dongle.push(build_frame(dongle.dongle, dongle.serial, 3, register, values))

    monkeypatch.setattr(dongle, "write", write)
    result = await client.async_write_register(FAN_REG, 1, mask=1)

    assert result is False
    assert client.get_cached_data()["hold"] == {}
    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, 0xC8C9), (3, FAN_REG, 1)]


@pytest.mark.asyncio
async def test_missing_fan_ack_does_not_retry_uncertain_write(fan_client):
    client, dongle = fan_client
    dongle.silent_functions = {6}

    result = await client.async_write_register(FAN_REG, 1, mask=1)

    assert result is False
    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, 0xC8C9)]
    assert dongle.hold[FAN_REG] == 0xC8C9  # The write took effect despite losing the ACK.
    assert client.get_cached_data()["hold"] == {}


@pytest.mark.parametrize("failure", ["unsupported", "exception", "short", "sentinel", "timeout"])
async def test_fan_write_requires_fresh_supported_firmware(fan_client, monkeypatch, failure):
    client, dongle = fan_client
    assert (await client.async_get_data())["hold"][7] == 0x6342
    dongle.requests.clear()
    if failure == "unsupported":
        dongle.hold[7] = 0x4142
    elif failure == "exception":
        dongle.exceptions[(3, 7)] = 2
    elif failure == "sentinel":
        dongle.hold[9] = 0xFFFF
    else:
        original_write = dongle.write

        def write(request):
            original_write(request)
            if dongle.requests[-1] == READ_FIRMWARE:
                dongle._pending.clear()
                if failure == "short":
                    dongle.push(build_frame(dongle.dongle, dongle.serial, 3, 7, [0x6342, 0x6161]))

        monkeypatch.setattr(dongle, "write", write)

    assert not await client.async_write_register(FAN_REG, 1, mask=1)
    assert dongle.requests == [READ_FIRMWARE]
    assert dongle.hold[FAN_REG] == 0xC8C8


async def test_unrelated_write_does_not_require_supported_firmware(fan_client):
    client, dongle = fan_client
    dongle.hold[7] = 0x4142

    assert await client.async_write_register(21, 0x200)
    assert dongle.requests == [(6, 21, 0x200), (3, 0, 125)]


@pytest.fixture
async def fan_setup(hass, enable_custom_integrations, monkeypatch):
    dongle = FanDongle()

    async def connect(*_args, **_kwargs):
        return dongle, dongle

    monkeypatch.setattr("asyncio.open_connection", connect)
    monkeypatch.setattr(
        "custom_components.lxp_modbus.classes.connection_manager.INITIAL_DATA_TIMEOUT", 0.01
    )
    monkeypatch.setattr("custom_components.lxp_modbus.classes.modbus_client.READ_TIMEOUT", 0.05)

    async def setup():
        entry = MockConfigEntry(domain=DOMAIN, title="Inverter", data={
            CONF_HOST: "192.0.2.1", CONF_PORT: 8000, CONF_DONGLE_SERIAL: "BA0000000A",
            CONF_INVERTER_SERIAL: "0000000001", CONF_POLL_INTERVAL: 60,
            CONF_ENTITY_PREFIX: "lxp", CONF_CONNECTION_RETRIES: 1,
            CONF_BATTERY_ENTITIES: "none", "read_only": False,
        })
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        dongle.requests.clear()
        return entry

    return dongle, setup


def fan_id(hass, control):
    platform, suffix = control
    return er.async_get(hass).async_get_entity_id(platform, DOMAIN, f"lxp_178_{suffix}")


def fan_state(hass, control):
    entity_id = fan_id(hass, control)
    assert entity_id is not None, f"Missing {control[1]} on Bcaa firmware"
    return hass.states.get(entity_id)


@pytest.mark.parametrize("control, expected, speeds", [
    (N1, 0xB5C7, (99, 90)),
    (N2, 0xC7C9, (100, 99)),
])
async def test_max_speed_writes_only_its_own_bits(hass, fan_setup, control, expected, speeds):
    dongle, setup = fan_setup
    await setup()
    target = fan_state(hass, control)
    assert float(target.state) == 100
    assert {key: target.attributes[key] for key in ("min", "max", "step", "unit_of_measurement")} == {
        "min": 20, "max": 100, "step": 1, "unit_of_measurement": "%",
    }

    # The app changed both slopes and fan 2's speed since HA last polled.
    dongle.hold[FAN_REG] = 0xB5C9
    await hass.services.async_call(
        "number", "set_value", {"entity_id": target.entity_id, "value": 99}, blocking=True
    )

    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, expected), (3, FAN_REG, 1)]
    assert dongle.hold[FAN_REG] == expected
    assert tuple(float(fan_state(hass, limit).state) for limit in (N1, N2)) == speeds


@pytest.mark.parametrize("control, initial, option, expected, slopes", [
    (S1, 0xB4C6, "High Speed", 0xB4C7, ("High Speed", "Default")),
    (S2, 0xB4C7, "High Speed", 0xB5C7, ("High Speed", "High Speed")),
    (S1, 0xB5C7, "Default", 0xB5C6, ("Default", "High Speed")),
    (S2, 0xB5C7, "Default", 0xB4C7, ("High Speed", "Default")),
])
async def test_slope_writes_only_its_own_bit(hass, fan_setup, control, initial, option, expected, slopes):
    dongle, setup = fan_setup
    await setup()
    target = fan_state(hass, control)
    assert target.state == "Default"
    assert target.attributes["options"] == ["Default", "High Speed"]

    dongle.hold[FAN_REG] = initial
    await hass.services.async_call(
        "select", "select_option", {"entity_id": target.entity_id, "option": option}, blocking=True
    )

    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, expected), (3, FAN_REG, 1)]
    assert dongle.hold[FAN_REG] == expected
    assert tuple(fan_state(hass, slope).state for slope in (S1, S2)) == slopes
    assert tuple(float(fan_state(hass, limit).state) for limit in (N1, N2)) == (99, 90)


@pytest.mark.parametrize("value", [19, 20.5, 101])
async def test_invalid_max_speed_does_not_touch_the_dongle(hass, fan_setup, value):
    dongle, setup = fan_setup
    await setup()

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "number", "set_value", {"entity_id": fan_id(hass, N1), "value": value}, blocking=True
        )

    assert not dongle.requests


async def test_unimplemented_word_makes_fan_controls_unavailable(hass, fan_setup):
    dongle, setup = fan_setup
    dongle.hold[FAN_REG] = 0xFFFF
    await setup()

    assert [fan_state(hass, control).state for control in CONTROLS] == ["unavailable"] * 4


@pytest.mark.parametrize("family", [(0x4142, 0x4241), (0, 0)], ids=["different", "unknown"])
async def test_other_firmware_does_not_expose_fan_controls(hass, fan_setup, family):
    dongle, setup = fan_setup
    dongle.hold[7], dongle.hold[8] = family
    await setup()

    assert [fan_id(hass, control) for control in CONTROLS] == [None] * 4


async def refresh_hold(hass, entry):
    runtime = hass.data[DOMAIN][entry.entry_id]
    runtime["api_client"].request_hold_refresh()
    await runtime["coordinator"].async_refresh()
    await hass.async_block_till_done()


async def test_fan_controls_discovered_on_later_valid_firmware_poll(hass, fan_setup):
    dongle, setup = fan_setup
    dongle.exceptions[(3, 0)] = 2  # Initial holding block 0 (firmware) fails; others work.
    entry = await setup()
    assert all(fan_id(hass, control) is None for control in CONTROLS)

    del dongle.exceptions[(3, 0)]
    await refresh_hold(hass, entry)
    assert [fan_state(hass, control).state for control in CONTROLS] == ["100", "100", "Default", "Default"]
    ids = [fan_id(hass, control) for control in CONTROLS]

    await refresh_hold(hass, entry)
    assert ids == [fan_id(hass, control) for control in CONTROLS]


async def test_existing_fan_controls_cannot_write_after_firmware_becomes_unsupported(hass, fan_setup):
    dongle, setup = fan_setup
    entry = await setup()
    target = fan_state(hass, S1)
    dongle.hold[7], dongle.hold[8] = 0x4142, 0x4241
    await refresh_hold(hass, entry)
    dongle.requests.clear()

    # Core quietly skips unavailable entities when dispatching entity services.
    await hass.services.async_call(
        "select", "select_option", {"entity_id": target.entity_id, "option": "High Speed"}, blocking=True
    )
    assert not dongle.requests
    assert [fan_state(hass, control).state for control in CONTROLS] == ["unavailable"] * 4


async def test_queued_fan_service_rechecks_firmware_after_poll(hass, fan_setup, monkeypatch):
    dongle, setup = fan_setup
    entry = await setup()
    target = fan_state(hass, S1)
    runtime = hass.data[DOMAIN][entry.entry_id]
    client = runtime["api_client"]
    coordinator = runtime["coordinator"]
    poll_reading_firmware = asyncio.Event()
    finish_poll = asyncio.Event()
    write_waiting = asyncio.Event()
    publish_poll = asyncio.Event()
    poll_completed = asyncio.Event()
    original_drain = dongle.drain
    acquire = client._lock.acquire
    update = coordinator._async_update_data

    async def drain():
        if dongle.requests[-1][:2] == (3, 0):
            poll_reading_firmware.set()
            await finish_poll.wait()
        await original_drain()

    async def acquire_and_signal():
        if client._lock.locked():
            write_waiting.set()
        return await acquire()

    async def update_before_publishing():
        data = await update()
        poll_completed.set()
        await publish_poll.wait()
        return data

    monkeypatch.setattr(dongle, "drain", drain)
    monkeypatch.setattr(client._lock, "acquire", acquire_and_signal)
    monkeypatch.setattr(coordinator, "_async_update_data", update_before_publishing)
    dongle.hold[7], dongle.hold[8] = 0x4142, 0x4241
    client.request_hold_refresh()
    poll = asyncio.create_task(coordinator.async_refresh())
    service = None
    try:
        await asyncio.wait_for(poll_reading_firmware.wait(), timeout=2)
        service = asyncio.create_task(hass.services.async_call(
            "select", "select_option", {"entity_id": target.entity_id, "option": "High Speed"}, blocking=True
        ))
        await asyncio.wait_for(write_waiting.wait(), timeout=2)
        assert fan_state(hass, S1).state == "Default"
        finish_poll.set()
        await asyncio.wait_for(poll_completed.wait(), timeout=2)
        assert client.get_cached_data()["hold"][7] == 0x4142
        assert coordinator.data["hold"][7] == 0x6342  # Poll data is not published yet.
        result, = await asyncio.wait_for(asyncio.gather(service, return_exceptions=True), timeout=2)
        assert [req for req in dongle.requests if req[0] == 6] == []
        assert dongle.requests[-1] == READ_FIRMWARE
        assert dongle.hold[FAN_REG] == 0xC8C8
        assert isinstance(result, HomeAssistantError)
    finally:
        finish_poll.set()
        publish_poll.set()
        await asyncio.gather(*(task for task in (poll, service) if task is not None), return_exceptions=True)

    assert fan_state(hass, S1).state == "unavailable"


@pytest.mark.parametrize("failure", ["exception", "timeout", "sentinel"])
async def test_unreadable_fresh_word_does_not_fall_back_to_cache(hass, fan_setup, monkeypatch, failure):
    dongle, setup = fan_setup
    await setup()
    if failure == "exception":
        dongle.exceptions[(3, FAN_REG)] = 2
    elif failure == "timeout":
        original_write = dongle.write

        def write(request):
            original_write(request)
            if dongle.requests[-1] == (3, FAN_REG, 1):
                dongle._pending.clear()

        monkeypatch.setattr(dongle, "write", write)
    else:
        dongle.hold[FAN_REG] = 0xFFFF

    with pytest.raises(HomeAssistantError, match="register 178"):
        await hass.services.async_call(
            "select", "select_option", {"entity_id": fan_id(hass, S1), "option": "High Speed"},
            blocking=True,
        )

    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1)]
    assert fan_state(hass, S1).state == "Default"


async def test_invalid_slope_option_does_not_touch_the_dongle(hass, fan_setup):
    dongle, setup = fan_setup
    await setup()

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            "select", "select_option", {"entity_id": fan_id(hass, S1), "option": "Turbo"}, blocking=True
        )

    assert not dongle.requests


async def test_mismatched_readback_fails_service_without_publishing(hass, fan_setup):
    dongle, setup = fan_setup
    await setup()
    # Echo the requested FC06 word but return 10% for Fan 1 on FC03 readback.
    dongle.readback_word = 0xC814
    with pytest.raises(HomeAssistantError, match="register 178"):
        await hass.services.async_call(
            "number", "set_value", {"entity_id": fan_id(hass, N1), "value": 20}, blocking=True
        )
    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, 0xC828), (3, FAN_REG, 1)]
    assert float(fan_state(hass, N1).state) == 100


async def test_write_confirms_target_bits_and_publishes_actual_sibling_bits(hass, fan_setup):
    dongle, setup = fan_setup
    await setup()
    dongle.readback_word = 0x1428  # Fan 1 accepted 20%; Fan 2 changed independently to 10%.
    await hass.services.async_call(
        "number", "set_value", {"entity_id": fan_id(hass, N1), "value": 20}, blocking=True
    )

    assert dongle.requests == [READ_FIRMWARE, (3, FAN_REG, 1), (6, FAN_REG, 0xC828), (3, FAN_REG, 1)]
    assert float(fan_state(hass, N1).state) == 20
    assert float(fan_state(hass, N2).state) == 10


async def test_concurrent_number_and_select_keep_both_changes(hass, fan_setup):
    dongle, setup = fan_setup
    await setup()
    await asyncio.gather(
        hass.services.async_call(
            "number", "set_value", {"entity_id": fan_id(hass, N1), "value": 99}, blocking=True
        ),
        hass.services.async_call(
            "select", "select_option", {"entity_id": fan_id(hass, S2), "option": "High Speed"},
            blocking=True,
        ),
    )

    read = (3, FAN_REG, 1)
    both = [READ_FIRMWARE, read, (6, FAN_REG, 0xC9C6), read]
    assert dongle.requests in (
        [READ_FIRMWARE, read, (6, FAN_REG, 0xC8C6), read] + both,
        [READ_FIRMWARE, read, (6, FAN_REG, 0xC9C8), read] + both,
    )
    assert dongle.hold[FAN_REG] == 0xC9C6
    assert float(fan_state(hass, N1).state) == 99
    assert fan_state(hass, S2).state == "High Speed"
