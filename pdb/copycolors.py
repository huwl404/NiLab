from chimerax.atomic import Residue, AtomsArg
from chimerax.core.commands import CmdDesc, register, StringArg


def copy_residue_colors(session, from_atoms, to_atoms=None, target='r'):
    """
    Copy colors from one set of atoms to another by matching residue numbers.
    Supported targets: 'r', 'a', 's', or 'all'.
    """

    # 1. Extract source residues and their ribbon colors
    from_res = from_atoms.unique_residues
    # Use ribbon_color as the source of truth for the transfer
    res_color_map = {res.number: res.ribbon_color for res in from_res}

    if not res_color_map:
        session.logger.warning("CopyColors: No source residues with color information found.")
        return

    # 2. Parse targets (support comma-separated values like 'r,a')
    target_list = [t.strip().lower() for t in target.split(',')]
    if 'all' in target_list:
        target_list = ['r', 'a', 's']

    # 3. Apply colors to destination
    to_res = to_atoms.unique_residues
    count = 0

    for res in to_res:
        if res.number in res_color_map:
            color = res_color_map[res.number]

            # Apply to Ribbon
            if 'r' in target_list:
                res.ribbon_color = color

            # Apply to Atoms (Sticks/Spheres)
            if 'a' in target_list:
                res.atoms.colors = color

            # Apply to Surface
            if 's' in target_list:
                res.atoms.surface_colors = color

            count += 1
    session.logger.info(f"CopyColors: Successfully transferred colors for {count} residues.")


def register_command(session):
    desc = CmdDesc(
        required=[('from_atoms', AtomsArg)],
        keyword=[('to_atoms', AtomsArg), ('target', StringArg)],
        required_arguments=['to_atoms'],
        synopsis='Copy colors between structures by residue number matching'
    )
    register('copycolors', desc, copy_residue_colors, logger=session.logger)


# Register the command in the session
register_command(session)