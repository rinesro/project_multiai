"use client";

import { Panel, SectionLabel, Chip } from "./ui";
import { FOKUS_OPTIMASI, type FokusOptimasi } from "@/lib/types";

export function OptimasiSection({
  intentUser,
  onChange,
}: {
  intentUser: FokusOptimasi[];
  onChange: (v: FokusOptimasi[]) => void;
}) {
  function toggle(opt: FokusOptimasi) {
    if (intentUser.includes(opt)) {
      onChange(intentUser.filter((o) => o !== opt));
    } else {
      onChange([...intentUser, opt]);
    }
  }

  return (
    <Panel>
      <SectionLabel nomor="03" title="Fokus Optimasi" desc="Boleh pilih lebih dari satu." />
      <div className="flex flex-wrap gap-2.5">
        {FOKUS_OPTIMASI.map((opt) => (
          <Chip key={opt} label={opt} active={intentUser.includes(opt)} onClick={() => toggle(opt)} />
        ))}
      </div>
    </Panel>
  );
}
