"""The agentic flow (agent/runtime.py) writes the app DB from two threads at
once: the agent's audit rows on a dedicated asyncio loop thread, and
Message.save on the Streamlit script thread. This guards the property that
flow depends on — concurrent writers against one SQLite file both succeed
(cross-thread connections are allowed and a writer waits for the lock instead
of raising "database is locked"). Mirrors the connect_args in orm/models.py.
"""

import threading

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

_Base = declarative_base()


class _Row(_Base):
    __tablename__ = "concurrent_rows"
    id = Column(Integer, primary_key=True)
    tag = Column(String, nullable=False)


def test_concurrent_writers_from_two_threads_all_commit(tmp_path):
    db = tmp_path / "concurrent.sqlite3"
    engine = create_engine(
        f"sqlite:///{db}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )
    _Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    writes_per_thread = 50
    errors: dict[str, BaseException] = {}
    barrier = threading.Barrier(2)

    def writer(tag: str) -> None:
        barrier.wait()  # maximize overlap / lock contention
        try:
            for _ in range(writes_per_thread):
                session = Session()
                try:
                    session.add(_Row(tag=tag))
                    session.commit()
                finally:
                    session.close()
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            errors[tag] = exc

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("loop", "script")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    with Session() as session:
        assert session.query(_Row).count() == 2 * writes_per_thread

    engine.dispose()
