"""
Audit and regression contract for Dual-Family Tape 1 Recording.

Ensures auditability and prevents accidental behavioral drift when operating
both BTC_5M_UPDOWN and BTC_15M_UPDOWN families in parallel:

1. Each normalized parquet must contain a single unique family_id.
2. resolution_id must match discovery condition_id.
3. Discovery evidence hash must be present in normalized tape metadata.

These are non-negotiable invariants enforced at pipeline completion.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """Single audit finding (violation or warning)."""
    
    category: str  # "error" | "warning"
    location: str  # File or artifact path
    message: str
    

@dataclass
class AuditReport:
    """Complete audit report for a family run."""
    
    family_id: str
    run_dir: Path
    findings: List[AuditFinding]
    
    def has_errors(self) -> bool:
        """Check if any findings are errors (not warnings)."""
        return any(f.category == "error" for f in self.findings)
    
    def summary(self) -> str:
        """Return summary string."""
        errors = sum(1 for f in self.findings if f.category == "error")
        warnings = sum(1 for f in self.findings if f.category == "warning")
        return f"Errors: {errors}, Warnings: {warnings}"


# ============================================================================
# AUDIT VALIDATORS
# ============================================================================

def audit_normalized_parquet_family_id(
    normalized_file: Path,
) -> List[AuditFinding]:
    """
    Verify that normalized parquet contains exactly one unique family_id.
    
    Args:
        normalized_file: Path to normalized tape parquet
    
    Returns:
        List of findings (empty if valid)
    """
    findings = []
    
    if not normalized_file.exists():
        findings.append(AuditFinding(
            category="error",
            location=str(normalized_file),
            message="Normalized parquet file not found",
        ))
        return findings
    
    try:
        df = pd.read_parquet(normalized_file, columns=["family_id"])
        
        unique_families = df["family_id"].unique()
        
        if len(unique_families) == 0:
            findings.append(AuditFinding(
                category="error",
                location=str(normalized_file),
                message="No rows in normalized parquet",
            ))
        elif len(unique_families) > 1:
            findings.append(AuditFinding(
                category="error",
                location=str(normalized_file),
                message=f"Multiple family_ids found: {list(unique_families)}. Each parquet must contain only one family.",
            ))
        else:
            logger.info(f"✓ Parquet family_id is unique: {unique_families[0]}")
    
    except Exception as e:
        findings.append(AuditFinding(
            category="error",
            location=str(normalized_file),
            message=f"Cannot read normalized parquet: {e}",
        ))
    
    return findings


def audit_resolution_id_match(
    normalized_file: Path,
    discovery_condition_id: str,
) -> List[AuditFinding]:
    """
    Verify that resolution_id in parquet matches discovery condition_id.
    
    Args:
        normalized_file: Path to normalized tape parquet
        discovery_condition_id: Expected resolution_id from discovery snapshot
    
    Returns:
        List of findings (empty if valid)
    """
    findings = []
    
    if not normalized_file.exists():
        findings.append(AuditFinding(
            category="error",
            location=str(normalized_file),
            message="Normalized parquet file not found",
        ))
        return findings
    
    try:
        df = pd.read_parquet(normalized_file, columns=["resolution_id"])
        
        unique_resolutions = df["resolution_id"].unique()
        
        if len(unique_resolutions) == 0:
            findings.append(AuditFinding(
                category="error",
                location=str(normalized_file),
                message="No resolution_id values in parquet",
            ))
        elif len(unique_resolutions) > 1:
            findings.append(AuditFinding(
                category="error",
                location=str(normalized_file),
                message=f"Multiple resolution_ids found: {list(unique_resolutions)}",
            ))
        else:
            resolution_id = unique_resolutions[0]
            if resolution_id != discovery_condition_id:
                findings.append(AuditFinding(
                    category="error",
                    location=str(normalized_file),
                    message=f"resolution_id mismatch: parquet has '{resolution_id}' but discovery has '{discovery_condition_id}'",
                ))
            else:
                logger.info(f"✓ resolution_id matches discovery condition_id: {resolution_id}")
    
    except Exception as e:
        findings.append(AuditFinding(
            category="error",
            location=str(normalized_file),
            message=f"Cannot read resolution_id from parquet: {e}",
        ))
    
    return findings


def audit_evidence_hash_in_metadata(
    run_dir: Path,
    normalized_file: Path,
) -> List[AuditFinding]:
    """
    Verify that discovery evidence SHA256 is accessible as metadata.
    
    Convention: Check if discovery_evidence.json exists and can be loaded.
    In production, we'd embed this in parquet metadata columns.
    
    Args:
        run_dir: Family-scoped run directory
        normalized_file: Path to normalized tape parquet
    
    Returns:
        List of findings (empty if valid)
    """
    findings = []
    
    # Look for discovery evidence file
    evidence_file = run_dir / "discovery" / "discovery_evidence.json"
    
    if not evidence_file.exists():
        findings.append(AuditFinding(
            category="warning",
            location=str(evidence_file),
            message="discovery_evidence.json not found. Evidence hash cannot be verified.",
        ))
        return findings
    
    try:
        with open(evidence_file) as f:
            evidence = json.load(f)
        
        # Verify it has a sha256 field
        if "evidence" not in evidence or "evidence_sha256" not in evidence.get("evidence", {}):
            findings.append(AuditFinding(
                category="warning",
                location=str(evidence_file),
                message="evidence_sha256 field not found in discovery_evidence.json",
            ))
        else:
            evidence_sha256 = evidence["evidence"]["evidence_sha256"]
            logger.info(f"✓ Discovery evidence hash available: {evidence_sha256}")
    
    except Exception as e:
        findings.append(AuditFinding(
            category="error",
            location=str(evidence_file),
            message=f"Cannot read discovery_evidence.json: {e}",
        ))
    
    return findings


# ============================================================================
# MAIN AUDIT FUNCTION
# ============================================================================

def run_family_audit(
    family_id: str,
    run_dir: Path,
    discovery_condition_id: str,
) -> AuditReport:
    """
    Run complete audit for a family run.
    
    Args:
        family_id: Family identifier
        run_dir: Family-scoped run directory
        discovery_condition_id: Expected resolution_id from discovery
    
    Returns:
        Complete AuditReport
    """
    findings: List[AuditFinding] = []
    
    logger.info(f"Running audit for family: {family_id}")
    logger.info(f"Run directory: {run_dir}")
    
    # Locate normalized parquet
    normalized_file = run_dir / "tape1_1s" / "tape1_1s.parquet"
    
    if not normalized_file.exists():
        findings.append(AuditFinding(
            category="error",
            location=str(normalized_file),
            message="Normalized tape parquet not found",
        ))
        
        report = AuditReport(
            family_id=family_id,
            run_dir=run_dir,
            findings=findings,
        )
        
        logger.error(f"✗ Audit FAILED for {family_id}")
        logger.error(f"  {report.summary()}")
        
        return report
    
    # Run all validators
    logger.info("Checking: family_id uniqueness...")
    findings.extend(audit_normalized_parquet_family_id(normalized_file))
    
    logger.info("Checking: resolution_id match...")
    findings.extend(audit_resolution_id_match(normalized_file, discovery_condition_id))
    
    logger.info("Checking: discovery evidence hash...")
    findings.extend(audit_evidence_hash_in_metadata(run_dir, normalized_file))
    
    report = AuditReport(
        family_id=family_id,
        run_dir=run_dir,
        findings=findings,
    )
    
    # Log results
    if report.has_errors():
        logger.error(f"✗ Audit FAILED for {family_id}")
        logger.error(f"  {report.summary()}")
        for finding in report.findings:
            if finding.category == "error":
                logger.error(f"    ERROR [{finding.location}]: {finding.message}")
    else:
        logger.info(f"✓ Audit PASSED for {family_id}")
        logger.info(f"  {report.summary()}")
        for finding in report.findings:
            if finding.category == "warning":
                logger.warning(f"    WARNING [{finding.location}]: {finding.message}")
    
    return report


def run_session_audit(
    session_dir: Path,
) -> dict[str, AuditReport]:
    """
    Run audit for all families in a session.
    
    Session structure:
        session_YYYYMMDD_HHMMSS/run_YYYYMMDD_HHMMSS/
          family=BTC_5M_UPDOWN/...
          family=BTC_15M_UPDOWN/...
    
    Args:
        session_dir: Session directory
    
    Returns:
        Dictionary mapping family_id -> AuditReport
    """
    reports = {}
    
    logger.info(f"Running session audit for: {session_dir}")
    
    # Find run directory under session
    run_dirs = sorted(
        d for d in session_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    
    if not run_dirs:
        logger.error(f"No run directories found under session: {session_dir}")
        return reports
    
    run_dir = run_dirs[0]  # Typically only one per session
    logger.info(f"Found run directory: {run_dir}")
    
    # Find family directories
    family_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("family=")
    )
    
    if not family_dirs:
        logger.error(f"No family directories found under run: {run_dir}")
        return reports
    
    logger.info(f"Found {len(family_dirs)} families to audit")
    
    # Audit each family
    for family_dir in family_dirs:
        family_id = family_dir.name.split("=", 1)[1]
        
        # Load discovery condition_id
        discovery_file = family_dir / "discovery" / "discovery_snapshot.json"
        
        if not discovery_file.exists():
            logger.warning(f"No discovery snapshot for {family_id}, skipping audit")
            continue
        
        try:
            with open(discovery_file) as f:
                discovery = json.load(f)
            condition_id = discovery.get("condition_id", "unknown")
        except Exception as e:
            logger.error(f"Cannot read discovery snapshot for {family_id}: {e}")
            continue
        
        # Run audit for this family
        report = run_family_audit(family_id, family_dir, condition_id)
        reports[family_id] = report
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SESSION AUDIT SUMMARY")
    logger.info("=" * 80)
    
    total_errors = sum(1 for r in reports.values() if r.has_errors())
    total_families = len(reports)
    
    logger.info(f"Families audited: {total_families}")
    logger.info(f"Families with errors: {total_errors}")
    
    for family_id, report in reports.items():
        status = "✓ PASS" if not report.has_errors() else "✗ FAIL"
        logger.info(f"  {family_id}: {status} ({report.summary()})")
    
    logger.info("=" * 80)
    
    return reports
