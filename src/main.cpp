#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/result.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>
#include <parquet/types.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Config {
    std::int64_t batches = 250;
    std::int64_t rows_per_batch = 1'000'000;
    std::int64_t row_group_size = 128'000;
    int zstd_level = 3;
    int brotli_level = 5;
    int gzip_level = 6;
    std::string output_dir = "/data/parquet/batches";
    std::string codec = "zstd";
    std::string profile = "mixed_v2";
    bool generate_all = false;
    std::string mode = "generate";
    bool deep_validate = false;
};

std::string getenv_or(std::string_view key, std::string fallback) {
    const char* value = std::getenv(std::string(key).c_str());
    return value == nullptr ? fallback : std::string(value);
}

std::int64_t env_i64(std::string_view key, std::int64_t fallback) {
    const auto value = getenv_or(key, "");
    if (value.empty()) return fallback;
    try { return std::stoll(value); }
    catch (...) { throw std::runtime_error("Invalid integer environment variable: " + std::string(key)); }
}

int env_int(std::string_view key, int fallback) { return static_cast<int>(env_i64(key, fallback)); }

bool env_bool(std::string_view key, bool fallback) {
    const auto value = getenv_or(key, "");
    if (value.empty()) return fallback;
    return value == "1" || value == "true" || value == "TRUE" || value == "yes";
}

Config load_config() {
    Config c;
    c.batches = env_i64("BATCHES", c.batches);
    c.rows_per_batch = env_i64("ROWS_PER_BATCH", c.rows_per_batch);
    c.row_group_size = env_i64("ROW_GROUP_SIZE", c.row_group_size);
    c.zstd_level = env_int("ZSTD_LEVEL", c.zstd_level);
    c.brotli_level = env_int("BROTLI_LEVEL", c.brotli_level);
    c.gzip_level = env_int("GZIP_LEVEL", c.gzip_level);
    c.output_dir = getenv_or("OUTPUT_DIR", c.output_dir);
    c.codec = getenv_or("CODEC", c.codec);
    c.profile = getenv_or("DATA_PROFILE", c.profile);
    c.generate_all = env_bool("GENERATE_ALL", c.generate_all);
    c.mode = getenv_or("MODE", c.mode);
    c.deep_validate = env_bool("DEEP_VALIDATE", c.deep_validate);
    if (c.batches <= 0 || c.rows_per_batch <= 0 || c.row_group_size <= 0)
        throw std::runtime_error("BATCHES, ROWS_PER_BATCH and ROW_GROUP_SIZE must be positive");
    if (c.profile != "mixed_v2" && c.profile != "legacy")
        throw std::runtime_error("DATA_PROFILE must be mixed_v2 or legacy");
    return c;
}

std::string format_mib(std::uintmax_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0) << " MiB";
    return out.str();
}

std::string format_gib(std::uintmax_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GiB";
    return out.str();
}

std::string codec_name(parquet::Compression::type codec) {
    switch (codec) {
        case parquet::Compression::UNCOMPRESSED: return "uncompressed";
        case parquet::Compression::SNAPPY: return "snappy";
        case parquet::Compression::GZIP: return "gzip";
        case parquet::Compression::BROTLI: return "brotli";
        case parquet::Compression::LZ4: return "lz4";
        case parquet::Compression::ZSTD: return "zstd";
        default: return "unknown";
    }
}

parquet::Compression::type parse_codec(const std::string& value) {
    if (value == "none" || value == "uncompressed") return parquet::Compression::UNCOMPRESSED;
    if (value == "snappy") return parquet::Compression::SNAPPY;
    if (value == "gzip") return parquet::Compression::GZIP;
    if (value == "brotli") return parquet::Compression::BROTLI;
    if (value == "lz4") return parquet::Compression::LZ4;
    if (value == "zstd") return parquet::Compression::ZSTD;
    throw std::runtime_error("Unsupported CODEC: " + value);
}

std::shared_ptr<arrow::Schema> expected_schema(const Config& c) {
    if (c.profile == "legacy") {
        return arrow::schema({
            arrow::field("id", arrow::int64(), false),
            arrow::field("event_time", arrow::timestamp(arrow::TimeUnit::MILLI), false),
            arrow::field("batch_id", arrow::int32(), false),
            arrow::field("category", arrow::utf8(), false),
            arrow::field("value", arrow::float64(), false),
            arrow::field("quantity", arrow::int32(), false),
            arrow::field("is_valid", arrow::boolean(), false),
        });
    }
    return arrow::schema({
        arrow::field("id", arrow::int64(), false),
        arrow::field("user_id", arrow::int64(), false),
        arrow::field("session_id", arrow::int64(), false),
        arrow::field("event_time", arrow::timestamp(arrow::TimeUnit::MILLI), false),
        arrow::field("event_date", arrow::date32(), false),
        arrow::field("batch_id", arrow::int32(), false),
        arrow::field("category", arrow::utf8(), false),
        arrow::field("country", arrow::utf8(), false),
        arrow::field("status", arrow::utf8(), false),
        arrow::field("device_id", arrow::utf8(), false),
        arrow::field("request_hash", arrow::uint64(), false),
        arrow::field("value", arrow::float64(), false),
        arrow::field("quantity", arrow::int32(), false),
        arrow::field("amount", arrow::float64(), false),
        arrow::field("is_valid", arrow::boolean(), false),
        arrow::field("optional_score", arrow::float32(), true),
        arrow::field("description", arrow::utf8(), false),
    });
}

// Fast deterministic mixer: no per-row random-device calls and no external RNG state.
std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

arrow::Result<std::shared_ptr<arrow::Table>> build_dataset(
    std::int64_t rows, std::int64_t global_row_start, std::int64_t batch_id, const Config& c) {
    const auto schema = expected_schema(c);
    arrow::Int64Builder id;
    arrow::Int64Builder user_id;
    arrow::Int64Builder session_id;
    arrow::TimestampBuilder event_time(arrow::timestamp(arrow::TimeUnit::MILLI), arrow::default_memory_pool());
    arrow::Date32Builder event_date;
    arrow::Int32Builder batch;
    arrow::StringBuilder category;
    arrow::StringBuilder country;
    arrow::StringBuilder status;
    arrow::StringBuilder device_id;
    arrow::UInt64Builder request_hash;
    arrow::DoubleBuilder value_builder;
    arrow::Int32Builder quantity;
    arrow::DoubleBuilder amount;
    arrow::BooleanBuilder is_valid;
    arrow::FloatBuilder optional_score;
    arrow::StringBuilder description;

    constexpr std::array<std::string_view, 8> categories = {
        "analytics", "finance", "iot", "media", "retail", "security", "telemetry", "web"};
    constexpr std::array<std::string_view, 16> countries = {
        "DE", "FR", "GB", "US", "CA", "JP", "CN", "IN", "BR", "AU", "ES", "IT", "NL", "SE", "PL", "CH"};
    constexpr std::array<std::string_view, 5> statuses = {"ok", "created", "updated", "retry", "failed"};
    constexpr std::array<std::string_view, 16> devices = {
        "android-01", "android-02", "android-03", "android-04",
        "ios-01", "ios-02", "ios-03", "ios-04",
        "web-01", "web-02", "web-03", "web-04",
        "edge-01", "edge-02", "edge-03", "edge-04"};
    constexpr std::array<std::string_view, 4> descriptions = {
        "normal analytics event",
        "customer activity update",
        "device telemetry record",
        "transaction processing event"};

    for (auto* builder : std::array<arrow::ArrayBuilder*, 17>{
             &id, &user_id, &session_id, &event_time, &event_date, &batch, &category,
             &country, &status, &device_id, &request_hash, &value_builder, &quantity, &amount,
             &is_valid, &optional_score, &description}) {
        ARROW_RETURN_NOT_OK(builder->Reserve(rows));
    }

    constexpr std::int64_t base_timestamp_ms = 1'704'067'200'000; // 2024-01-01 UTC
    constexpr std::int32_t base_date = 19'723; // 2024-01-01 in days since epoch

    for (std::int64_t i = 0; i < rows; ++i) {
        const std::uint64_t g = static_cast<std::uint64_t>(global_row_start + i);
        const auto category_index = static_cast<std::size_t>((g * 13ULL) % categories.size());
        const auto country_index = static_cast<std::size_t>((g * 17ULL) % countries.size());
        const auto status_index = static_cast<std::size_t>((g * 19ULL) % statuses.size());
        const auto device_index = static_cast<std::size_t>(mix64(g) % devices.size());
        const auto description_index = static_cast<std::size_t>((g / 1000ULL) % descriptions.size());
        const auto quantity = static_cast<std::int32_t>((g * 17ULL) % 1000ULL);
        const auto value = 100.0 + static_cast<double>((g * 37ULL) % 10'000ULL) / 100.0;
        const auto amount_value = value * (1.0 + static_cast<double>(quantity) / 1000.0);
        const auto uid = static_cast<std::int64_t>((g * 7919ULL) % 10'000'000ULL);
        const auto sid = static_cast<std::int64_t>((g / 20ULL) % 50'000'000ULL);
        const auto request = mix64(g ^ 0xD1B54A32D192ED03ULL);
        const auto day_offset = static_cast<std::int32_t>((g / 86'400ULL) % 365ULL);

        ARROW_RETURN_NOT_OK(id.Append(static_cast<std::int64_t>(g)));
        ARROW_RETURN_NOT_OK(user_id.Append(uid));
        ARROW_RETURN_NOT_OK(session_id.Append(sid));
        ARROW_RETURN_NOT_OK(event_time.Append(base_timestamp_ms + static_cast<std::int64_t>(g) * 1'000));
        ARROW_RETURN_NOT_OK(event_date.Append(base_date + day_offset));
        ARROW_RETURN_NOT_OK(batch.Append(static_cast<std::int32_t>(batch_id)));
        ARROW_RETURN_NOT_OK(category.Append(categories[category_index]));
        ARROW_RETURN_NOT_OK(country.Append(countries[country_index]));
        ARROW_RETURN_NOT_OK(status.Append(statuses[status_index]));
        ARROW_RETURN_NOT_OK(device_id.Append(devices[device_index]));
        ARROW_RETURN_NOT_OK(request_hash.Append(request));
        ARROW_RETURN_NOT_OK(value_builder.Append(value));
        ARROW_RETURN_NOT_OK(quantity.Append(quantity));
        ARROW_RETURN_NOT_OK(amount.Append(amount_value));
        ARROW_RETURN_NOT_OK(is_valid.Append((g % 3ULL) != 0ULL));
        if ((g % 10ULL) == 0ULL) ARROW_RETURN_NOT_OK(optional_score.AppendNull());
        else ARROW_RETURN_NOT_OK(optional_score.Append(static_cast<float>((g * 29ULL) % 10000ULL) / 100.0F));
        ARROW_RETURN_NOT_OK(description.Append(descriptions[description_index]));
    }

    std::vector<std::shared_ptr<arrow::Array>> arrays;
    arrays.reserve(17);
    for (auto* builder : std::array<arrow::ArrayBuilder*, 17>{
             &id, &user_id, &session_id, &event_time, &event_date, &batch, &category,
             &country, &status, &device_id, &request_hash, &value_builder, &quantity, &amount,
             &is_valid, &optional_score, &description}) {
        std::shared_ptr<arrow::Array> array;
        ARROW_RETURN_NOT_OK(builder->Finish(&array));
        arrays.push_back(std::move(array));
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
    columns.reserve(arrays.size());
    for (auto& array : arrays) columns.push_back(std::make_shared<arrow::ChunkedArray>(std::move(array)));
    return arrow::Table::Make(schema, columns, rows);
}

std::shared_ptr<parquet::WriterProperties> make_properties(parquet::Compression::type codec, const Config& c) {
    parquet::WriterProperties::Builder builder;
    builder.compression(codec)
        ->enable_dictionary()
        ->data_pagesize(1024 * 1024)
        ->dictionary_pagesize_limit(1024 * 1024)
        ->max_rows_per_page(20'000)
        ->max_row_group_length(c.row_group_size)
        ->version(parquet::ParquetVersion::PARQUET_2_6);
    switch (codec) {
        case parquet::Compression::ZSTD: builder.compression_level(c.zstd_level); break;
        case parquet::Compression::BROTLI: builder.compression_level(c.brotli_level); break;
        case parquet::Compression::GZIP: builder.compression_level(c.gzip_level); break;
        default: break;
    }
    return builder.build();
}

arrow::Result<std::uintmax_t> write_parquet(const std::shared_ptr<arrow::Table>& table,
                                             const fs::path& output_path,
                                             parquet::Compression::type codec,
                                             const Config& c,
                                             std::int64_t batch_number,
                                             std::int64_t total_batches) {
    const auto started = Clock::now();
    ARROW_ASSIGN_OR_RAISE(auto output, arrow::io::FileOutputStream::Open(output_path.string()));
    const auto properties = make_properties(codec, c);
    ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), output,
                                                     c.row_group_size, properties));
    ARROW_RETURN_NOT_OK(output->Close());
    const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const auto bytes = fs::file_size(output_path);
    const auto mib_per_second = static_cast<double>(bytes) / (1024.0 * 1024.0) / elapsed;
    std::cout << "[batch] " << std::setw(3) << batch_number << '/' << total_batches
              << " codec=" << codec_name(codec) << " rows=" << table->num_rows()
              << " size=" << format_mib(bytes) << " write=" << std::fixed << std::setprecision(3)
              << elapsed << " s throughput=" << std::setprecision(1) << mib_per_second
              << " MiB/s file=" << output_path.filename().string() << '\n';
    return bytes;
}

arrow::Status validate_file(const fs::path& path, parquet::Compression::type expected_codec,
                            const Config& c, std::int64_t expected_batch) {
    if (!fs::exists(path) || fs::file_size(path) < 8)
        return arrow::Status::IOError("invalid parquet file: ", path.string());
    std::unique_ptr<parquet::ParquetFileReader> raw_reader;
    try { raw_reader = parquet::ParquetFileReader::OpenFile(path.string(), false); }
    catch (const std::exception& ex) { return arrow::Status::Invalid("cannot open parquet footer: ", ex.what()); }
    const auto metadata = raw_reader->metadata();
    if (!metadata) return arrow::Status::Invalid("missing parquet metadata: ", path.string());
    if (metadata->num_columns() != static_cast<int>(expected_schema(c)->num_fields()))
        return arrow::Status::Invalid("unexpected column count: ", path.string());
    if (metadata->num_rows() != c.rows_per_batch)
        return arrow::Status::Invalid("unexpected row count: ", path.string());
    const auto expected_row_groups = (c.rows_per_batch + c.row_group_size - 1) / c.row_group_size;
    if (metadata->num_row_groups() != expected_row_groups)
        return arrow::Status::Invalid("unexpected row-group count: ", path.string());
    std::shared_ptr<arrow::Schema> schema;
    try {
        ARROW_ASSIGN_OR_RAISE(auto readable, arrow::io::ReadableFile::Open(path.string()));
        std::unique_ptr<parquet::arrow::FileReader> reader;
        ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(readable, arrow::default_memory_pool()));
        ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
    } catch (const std::exception& ex) { return arrow::Status::Invalid("cannot read schema: ", ex.what()); }
    if (!schema->Equals(*expected_schema(c), false)) return arrow::Status::Invalid("schema mismatch: ", path.string());
    std::int64_t accumulated = 0;
    for (int rg = 0; rg < metadata->num_row_groups(); ++rg) {
        const auto row_group = metadata->RowGroup(rg);
        if (row_group->num_rows() <= 0 || row_group->num_rows() > c.row_group_size)
            return arrow::Status::Invalid("invalid row-group size: ", path.string());
        accumulated += row_group->num_rows();
        for (int col = 0; col < row_group->num_columns(); ++col) {
            const auto column = row_group->ColumnChunk(col);
            if (column->num_values() != row_group->num_rows()) return arrow::Status::Invalid("column value count mismatch");
            if (column->compression() != expected_codec) return arrow::Status::Invalid("unexpected compression codec");
            if (!column->is_stats_set()) return arrow::Status::Invalid("missing statistics");
        }
    }
    if (accumulated != c.rows_per_batch) return arrow::Status::Invalid("row-group row count mismatch");
    if (c.deep_validate) {
        ARROW_ASSIGN_OR_RAISE(auto readable, arrow::io::ReadableFile::Open(path.string()));
        std::unique_ptr<parquet::arrow::FileReader> reader;
        ARROW_ASSIGN_OR_RAISE(reader, parquet::arrow::OpenFile(readable, arrow::default_memory_pool()));
        std::shared_ptr<arrow::ChunkedArray> ids;
        std::shared_ptr<arrow::ChunkedArray> batch_ids;
        ARROW_RETURN_NOT_OK(reader->ReadColumn(0, &ids));
        ARROW_RETURN_NOT_OK(reader->ReadColumn(5, &batch_ids));
        std::int64_t expected_id = (expected_batch - 1) * c.rows_per_batch;
        for (int chunk = 0; chunk < ids->num_chunks(); ++chunk) {
            auto id_array = std::static_pointer_cast<arrow::Int64Array>(ids->chunk(chunk));
            auto batch_array = std::static_pointer_cast<arrow::Int32Array>(batch_ids->chunk(chunk));
            for (int64_t i = 0; i < id_array->length(); ++i, ++expected_id) {
                if (id_array->Value(i) != expected_id || batch_array->Value(i) != expected_batch)
                    return arrow::Status::Invalid("deep validation mismatch");
            }
        }
    }
    return arrow::Status::OK();
}

arrow::Status validate_codec(const Config& c, parquet::Compression::type codec) {
    const auto dir = fs::path(c.output_dir) / codec_name(codec);
    if (!fs::exists(dir)) return arrow::Status::IOError("missing codec directory: ", dir.string());
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.is_regular_file() && entry.path().extension() == ".parquet") files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    if (static_cast<std::int64_t>(files.size()) != c.batches) return arrow::Status::Invalid("unexpected parquet file count");
    const auto started = Clock::now();
    std::uintmax_t total_bytes = 0;
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(files.size()); ++i) {
        std::ostringstream expected;
        expected << "batch_" << std::setfill('0') << std::setw(6) << (i + 1) << ".parquet";
        if (files[i].filename().string() != expected.str()) return arrow::Status::Invalid("unexpected batch filename");
        ARROW_RETURN_NOT_OK(validate_file(files[i], codec, c, i + 1));
        total_bytes += fs::file_size(files[i]);
        if (c.batches <= 20 || (i + 1) % std::max<std::int64_t>(1, c.batches / 20) == 0 || i + 1 == c.batches)
            std::cout << "[check] " << std::setw(3) << i + 1 << '/' << c.batches << " ok\n";
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    std::cout << "[check] codec=" << codec_name(codec) << " files=" << c.batches
              << " rows=" << c.batches * c.rows_per_batch << " size=" << format_gib(total_bytes)
              << " elapsed=" << std::fixed << std::setprecision(3) << elapsed
              << " s mode=" << (c.deep_validate ? "deep" : "metadata") << '\n';
    return arrow::Status::OK();
}

arrow::Status run_codec(const Config& c, parquet::Compression::type codec) {
    const auto dir = fs::path(c.output_dir) / codec_name(codec);
    fs::create_directories(dir);
    std::uintmax_t total_bytes = 0;
    const auto all_started = Clock::now();
    std::cout << "\n[run  ] profile=" << c.profile << " codec=" << codec_name(codec)
              << " batches=" << c.batches << " rows_per_batch=" << c.rows_per_batch
              << " total_rows=" << c.batches * c.rows_per_batch
              << " row_group_size=" << c.row_group_size << "\n";
    for (std::int64_t batch = 1; batch <= c.batches; ++batch) {
        const auto global_start = (batch - 1) * c.rows_per_batch;
        const auto gen_started = Clock::now();
        ARROW_ASSIGN_OR_RAISE(const auto table, build_dataset(c.rows_per_batch, global_start, batch, c));
        const auto gen_elapsed = std::chrono::duration<double>(Clock::now() - gen_started).count();
        std::ostringstream name;
        name << "batch_" << std::setfill('0') << std::setw(6) << batch << ".parquet";
        ARROW_ASSIGN_OR_RAISE(const auto bytes, write_parquet(table, dir / name.str(), codec, c, batch, c.batches));
        total_bytes += bytes;
        std::cout << "        generated=" << std::fixed << std::setprecision(3) << gen_elapsed
                  << " s cumulative_size=" << format_gib(total_bytes) << '\n';
    }
    const auto elapsed = std::chrono::duration<double>(Clock::now() - all_started).count();
    std::cout << "[total ] codec=" << codec_name(codec) << " files=" << c.batches
              << " rows=" << c.batches * c.rows_per_batch << " size=" << format_gib(total_bytes)
              << " elapsed=" << std::fixed << std::setprecision(3) << elapsed
              << " s throughput=" << static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / elapsed << " GiB/s\n";
    return arrow::Status::OK();
}

arrow::Status run_generation(const Config& c) {
    if (c.generate_all) {
        for (const auto codec : {"uncompressed", "snappy", "zstd", "brotli"})
            ARROW_RETURN_NOT_OK(run_codec(c, parse_codec(codec)));
    } else {
        ARROW_RETURN_NOT_OK(run_codec(c, parse_codec(c.codec)));
    }
    return arrow::Status::OK();
}

arrow::Status run_validation(const Config& c) {
    std::cout << "\n[check] profile=" << c.profile << " starting parquet structure validation\n";
    if (c.generate_all) {
        for (const auto codec : {"uncompressed", "snappy", "zstd", "brotli"})
            ARROW_RETURN_NOT_OK(validate_codec(c, parse_codec(codec)));
    } else {
        ARROW_RETURN_NOT_OK(validate_codec(c, parse_codec(c.codec)));
    }
    return arrow::Status::OK();
}

int main() {
    try {
        const auto config = load_config();
        std::cout << "[config] profile=" << config.profile << " batches=" << config.batches
                  << " rows_per_batch=" << config.rows_per_batch << " row_group_size=" << config.row_group_size
                  << " codec=" << config.codec << "\n";
        if (config.mode == "generate") ARROW_RETURN_NOT_OK(run_generation(config));
        else if (config.mode == "validate") ARROW_RETURN_NOT_OK(run_validation(config));
        else if (config.mode == "generate_and_validate") {
            ARROW_RETURN_NOT_OK(run_generation(config));
            ARROW_RETURN_NOT_OK(run_validation(config));
        } else throw std::runtime_error("Unsupported MODE: " + config.mode);
        return 0;
    } catch (const arrow::Status& status) {
        std::cerr << "[error] " << status.ToString() << '\n';
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << '\n';
        return 1;
    }
}
