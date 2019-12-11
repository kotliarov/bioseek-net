""" make_dataset.py
    Collection of utils that make 
"""
import ftplib, gzip, re, tempfile
from pathlib import Path
import shutil, Bio
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pandas as pd

class Dataset(object):

    def __init__(self):
        pass

    def load(self, dataset='all'):
        if dataset == 'all':
            filenames = [
             'uniprot_sprot.fasta.gz', 'uniprot_trembl.fasta.gz']
        elif dataset == 'sprot':
            filenames = [
             'uniprot_sprot.fasta.gz']
        elif dataset == 'trembl':
            filenames = [
             'uniprot_trembl.fasta.gz']
        else:
            raise ValueError('Expected dataset name ["all", "sprot", "trembl"]')

        def get_release_name(filepath):
            """ Return name of uniprot release.
            """
            with open(filepath, 'r') as (src):
                match = re.search('(?m)(?<=<version>)(\\d{4}_\\d\\d?)(?=</version>)', src.read())
                return match.groups()[0]

        with tempfile.TemporaryDirectory() as (workdir):
            print('Workdir={}'.format(workdir))
            with Path(workdir) as (folder):
                ftp = ftplib.FTP('ftp.uniprot.org')
                ftp.login()
                ftp.cwd('/pub/databases/uniprot/current_release/knowledgebase/complete')

                def download(filename):
                    with open(folder / filename, 'wb') as (dest):
                        ftp.retrbinary('RETR {}'.format(filename), dest.write)
                    return folder / filename

                release = get_release_name(download('RELEASE.metalink'))
                paths = []
                for filename in filenames:
                    paths.append(download(filename))

                ftp.quit()
                inflated = []
                for index, path in enumerate(paths):
                    with gzip.open(path, 'rb') as (src):
                        inflated.append(folder / 'dataset_{}.fa'.format(index))
                        with open(inflated[(-1)], 'wb') as (dest):
                            shutil.copyfileobj(src, dest)

                return (
                 release, self._make_dataframe(inflated))

    def _make_dataframe(self, filepaths: []):
        """ Return instance of panda's DataFrame
        of following structure:
            - ID
            - name
            - dataset
            - organism
            - accession
            - defline
            - sequence
        """
        rows = []
        dataset_dict = {'sp':'Swiss-Prot', 
         'tr':'TrEMBL'}
        rex_pe = re.compile('(?<=PE=)\\d')
        rex_organism = re.compile('(?<=OS=)(.*?) OX=')
        for filepath in filepaths:
            for seq_record in SeqIO.parse(filepath, 'fasta'):
                sid = seq_record.id.split('|')
                accession = sid[1]
                dataset = dataset_dict[sid[0]]
                name = sid[2]
                description = seq_record.description
                sequence = str(seq_record.seq)
                m = rex_pe.search(description)
                pe = int(m.group(0))
                m = rex_organism.search(description)
                organism = m.groups()[0]
                data_dict = {'ID':accession, 
                 'name':name,  'dataset':dataset,  'proteinexistence':pe,  'organism':organism,  'sequence':sequence}
                rows.append(data_dict)

        df = pd.DataFrame(rows).set_index('ID')
        df['name'] = df.name.astype(str)
        df['dataset'] = df.dataset.astype('category')
        df['organism'] = df.organism.astype('category')
        df['sequence'] = df.sequence.astype(str)
        return df
