export function createBus() {
  const map = new Map();
  return {
    on(type, fn) {
      const list = map.get(type) ?? [];
      list.push(fn);
      map.set(type, list);
      return () => {
        const current = map.get(type) ?? [];
        map.set(type, current.filter((item) => item !== fn));
      };
    },
    emit(type, payload) {
      for (const fn of map.get(type) ?? []) fn(payload);
    },
  };
}
