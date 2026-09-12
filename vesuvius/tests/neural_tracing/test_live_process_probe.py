from vesuvius.neural_tracing.fiber_trace_3d.live_process_probe import tcp_counters


def test_tcp_identity_ignores_changing_queue_occupancy():
    def sample(queue, received):
        return tcp_counters(
            f'ESTAB {queue} 0 10.0.0.1:100 10.0.0.2:443 users:(("python",pid=42,fd=8))\n'
            f'\t cubic bytes_sent:20 bytes_received:{received}\n', {42})
    first, second = sample(100, 200), sample(0, 300)
    assert first.keys() == second.keys()
    key = next(iter(first))
    assert second[key][1] - first[key][1] == 100


def test_tcp_ignores_unrelated_and_ownerless_sockets():
    text = ('ESTAB 0 0 a:1 b:2 users:(("python",pid=42,fd=8))\n'
            '\t bytes_received:100\n'
            'ESTAB 0 0 a:3 b:4 users:(("other",pid=43,fd=9))\n'
            '\t bytes_received:900\n'
            'ESTAB 0 0 a:5 b:6\n'
            '\t bytes_received:999\n')
    assert list(tcp_counters(text, {42}).values()) == [(42, 100, 0)]


def test_futex_wait_addresses_are_architecture_and_operation_specific():
    from vesuvius.neural_tracing.fiber_trace_3d.live_process_probe import futex_wait_address
    assert futex_wait_address("202 0x1234 0x80 0 0", "x86_64") == "0x1234"
    assert futex_wait_address("98 0xabcd 0x89 0 0", "aarch64") == "0xabcd"
    assert futex_wait_address("202 0x1234 0x81 0 0", "x86_64") is None  # WAKE
    assert futex_wait_address("449 0x1234 0 0 0", "x86_64") is None  # futex_waitv
    assert futex_wait_address("202 0x1234 0x80 0 0", "aarch64") is None
    assert futex_wait_address("running", "x86_64") is None
    assert futex_wait_address("202 0x1234 0x80 0 0", "unknown") is None
