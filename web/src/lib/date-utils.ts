const EPOCH = new Date(0);
const MS_PER_DAY = 86400000;

export const dateToDaysFromToday = (dateStr: string): number => {
  const target = new Date(dateStr);
  target.setHours(0, 0, 0, 0);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.floor((target.getTime() - today.getTime()) / MS_PER_DAY);
};

export const dateToIndex = (dateStr: string): number => {
  const d = new Date(dateStr);
  return Math.floor((d.getTime() - EPOCH.getTime()) / MS_PER_DAY);
};

export const todayIndex = (): number => {
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.floor((now.getTime() - EPOCH.getTime()) / MS_PER_DAY);
};

export { EPOCH, MS_PER_DAY };
