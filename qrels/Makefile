all: web robust deep-docs deep-pass core legal ml-1M beerAdvocate libraryThing amzn-cds-vinyl amzn-musical-instruments amzn-software amzn-video-games

web:
	mkdir web
	cd web; curl -s -o 2009a.txt.gz https://trec.nist.gov/data/web/09/qrels.diversity.gz; cat 2009a.txt.gz | gunzip > 2009a.txt; ../collapse-diversity-data.sh 2009a.txt > 2009.txt; rm -f 2009a.txt 2009a.txt.gz
	cd web; curl -s -o 2010a.txt https://trec.nist.gov/data/web/10/10.diversity-qrels.final; ../collapse-diversity-data.sh 2010a.txt > 2010.txt; rm -f 2010a.txt
	cd web; curl -s -o 2011a.txt https://trec.nist.gov/data/web/11/qrels.diversity; ../collapse-diversity-data.sh 2011a.txt > 2011.txt; rm -f 2011a.txt
	cd web; curl -s -o 2012a.txt https://trec.nist.gov/data/web/12/qrels.diversity; ../collapse-diversity-data.sh 2012a.txt > 2012.txt; rm -f 2012a.txt
	cd web; curl -s -o 2013a.txt https://trec.nist.gov/data/web/2013/qrels.all.txt; ../collapse-diversity-data.sh 2013a.txt > 2013.txt; rm -f 2013a.txt
	cd web; curl -s -o 2014a.txt https://trec.nist.gov/data/web/2014/qrels.all.txt; ../collapse-diversity-data.sh 2014a.txt > 2014.txt; rm -f 2014a.txt

robust:
	mkdir robust
	cd robust; curl -s -o 2004.txt https://trec.nist.gov/data/robust/qrels.robust2004.txt

core:
	mkdir core
	cd core; curl -s -o 2018.txt https://trec.nist.gov/data/core/qrels2018.txt
	cd core; curl -s -o 2017.txt https://trec.nist.gov/data/core/qrels.txt

deep-docs: 
	mkdir deep-docs
	cd deep-docs; curl -s -o 2023.txt https://trec.nist.gov/data/deep/2023.qrels.docs.withDupes.txt
	cd deep-docs; curl -s -o 2022.txt https://trec.nist.gov/data/deep/2022.qrels.docs.inferred.txt
	cd deep-docs; curl -s -o 2021.txt https://trec.nist.gov/data/deep/2021.qrels.docs.expanded.txt
	cd deep-docs; curl -s -o 2020.txt https://trec.nist.gov/data/deep/2020qrels-docs.txt
	cd deep-docs; curl -s -o 2019.txt https://trec.nist.gov/data/deep/2019qrels-docs.txt

deep-pass: 
	mkdir deep-pass
	cd deep-pass; curl -s -o 2023.txt https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt
	cd deep-pass; curl -s -o 2022.txt https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt
	cd deep-pass; curl -s -o 2021.txt https://trec.nist.gov/data/deep/2021.qrels.pass.expanded.txt
	cd deep-pass; curl -s -o 2020.txt https://trec.nist.gov/data/deep/2020qrels-pass.txt
	cd deep-pass; curl -s -o 2019.txt https://trec.nist.gov/data/deep/2019qrels-pass.txt
	
legal:
	mkdir legal
	cd legal; curl -s -o 2006.txt https://trec.nist.gov/data/legal/06/qrels.legal06
	cd legal; curl -s -o 2007.txt https://trec.nist.gov/data/legal/07/qrelsL07.normal
	# cd legal; curl -s -O https://trec.nist.gov/data/legal/08/resultsL08.zip; unzip -p resultsL08.zip qrelsL08.normal > 2008.txt; rm -f resultsL08.zip
	# cd legal; curl -s -O https://trec.nist.gov/data/legal/09/resultsL09.zip; unzip -p resultsL09.zip qrelsL09.normal > 2009.txt; rm -f resultsL09.zip

ml-1M:
	mkdir ml-1M
	cd ml-1M; curl -s -L https://github.com/dvalcarce/evalMetrics/raw/master/qrels/qrels-ml-1M/qrels-1.00-sample000/u1.qrel.gz | gunzip > 2018.txt

beerAdvocate:
	mkdir beerAdvocate
	cd beerAdvocate; curl -s -L https://github.com/dvalcarce/evalMetrics/raw/master/qrels/qrels-beerAdvocate/qrels-1.00-sample000/u1.qrel.gz | gunzip > 2018.txt

libraryThing:
	mkdir libraryThing
	cd libraryThing; curl -s -L https://github.com/dvalcarce/evalMetrics/raw/master/qrels/qrels-libraryThing/qrels-1.00-sample000/u1.qrel.gz | gunzip > 2018.txt

amzn-cds-vinyl:
	mkdir $@
	cd $@;  curl `awk '{print $$2}' ../../urls.txt |grep qrels|grep test|grep CD` | gunzip -c | tr "\t" " " > 2023.txt
	cp -R $@ $@-100
	cp -R $@ $@-1000
	cp -R $@ $@-2000
amzn-musical-instruments:
	mkdir $@
	cd $@;  curl `awk '{print $$2}' ../../urls.txt |grep qrels|grep test|grep Musical_Instruments` | gunzip -c | tr "\t" " " > 2023.txt
	cp -R $@ $@-100
	cp -R $@ $@-1000
	cp -R $@ $@-2000
amzn-software:
	mkdir $@
	cd $@;  curl `awk '{print $$2}' ../../urls.txt |grep qrels|grep test|grep Software` | gunzip -c | tr "\t" " " > 2023.txt
	cp -R $@ $@-100
	cp -R $@ $@-1000
	cp -R $@ $@-2000
amzn-video-games:
	mkdir $@
	cd $@;  curl `awk '{print $$2}' ../../urls.txt |grep qrels|grep test|grep Video_Games` | gunzip -c | tr "\t" " " > 2023.txt
	cp -R $@ $@-100
	cp -R $@ $@-1000
	cp -R $@ $@-2000
clean:
	rm -Rf robust deep-docs deep-pass core legal web ml-1M beerAdvocate libraryThing amzn-*